import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotbext.axi import AxiStreamSource, AxiStreamSink, AxiStreamBus
from cocotb_bus.bus import Bus
from chop.nn.dwn.model import DWNModel
from mase_components.dwn_layers.hardware_core import DWNHardwareCore
import torch
import os
import numpy as np
import math
import random

model_path = os.environ['MODEL_PATH']
model_mode = os.environ['MODEL_MODE']

def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1), bitorder="little").tobytes()

def uint8_to_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype=np.uint8).reshape(-1).tobytes()

def bytes_to_bits(payload, n_bits: int) -> np.ndarray:
    byte_count = math.ceil(n_bits / 8)
    raw_bytes = bytes(int(x) & 0xFF for x in payload)
    byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8, count=byte_count)
    return np.unpackbits(byte_arr, bitorder="little")[:n_bits]

def class_scores_to_bits(class_scores: np.ndarray, total_bits: int) -> np.ndarray:
    scores = np.asarray(class_scores, dtype=np.int64).reshape(-1)
    num_classes = scores.size
    assert num_classes != 0
    assert total_bits % num_classes == 0
    class_width = total_bits // num_classes
    scores = scores.astype(np.uint64)[::-1]
    bit_idx = np.arange(class_width, dtype=np.uint64)
    bits = ((scores[:, None] >> bit_idx) & 1).astype(np.uint8).reshape(-1)
    return bits

class AxiStreamBusUncooked(AxiStreamBus):
    def __init__(self, entity, prefix):
        signals = {'tdata': prefix}
        optional_signals = {
            'tvalid': f'{prefix}_valid',
            'tready': f'{prefix}_ready',
        # These are not used by MASE generated RTL but game is game
            'tlast': f'{prefix}_tlast',
            'tkeep': f'{prefix}_tkeep',
            'tid': f'{prefix}_tid',
            'tdest': f'{prefix}_tdest',
            'tuser': f'{prefix}_tuser',
        }
        Bus.__init__(self, entity, None, signals, optional_signals)

class TopEnv():
    def __init__(self, dut):
        self.dut = dut
        self.mode = os.environ['MODEL_MODE']
        model = torch.load(model_path, weights_only=True, map_location='cpu')
        cfg = model['model_config']
        model_kwargs = {k: v for k, v in cfg.items() if k not in ("area_lambda", "lambda_reg")}
        self.full_model = DWNModel(**model_kwargs)
        self.full_model.fit_thermometer(torch.zeros(2, cfg["input_features"]))
        self.full_model.load_state_dict(model['model_state_dict'])
        self.full_model.eval()
        self.core_model = DWNHardwareCore(self.full_model.lut_layers)
        self.core_model.eval()
        self.model_cfg = cfg
        self.input_bit_width = len(dut.data_in_0)
        self.output_bit_width = len(dut.data_out_0)
        self.clock = Clock(dut.clk, 10, units='ns')
        self.axis_source = AxiStreamSource(AxiStreamBusUncooked(dut, "data_in_0"), dut.clk, dut.rst)
        self.axis_sink = AxiStreamSink(AxiStreamBusUncooked(dut, "data_out_0"), dut.clk, dut.rst)

    async def reset_dut(self):
        self.dut.rst.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.rst.value = 0

    async def start(self):
        cocotb.start_soon(self.clock.start())
        await self.reset_dut()

    def get_random_input(self, batch_size=1) -> torch.Tensor:
        if self.mode == 'core':
            return torch.randint(0, 2, (batch_size, self.input_bit_width), dtype=torch.uint8)
        else:
            return torch.randint(0, 256, (batch_size, self.model_cfg["input_features"]), dtype=torch.uint8)

    def encode_input_data(self, input_tensor: torch.Tensor) -> bytes:
        data = input_tensor.cpu().numpy()
        return bits_to_bytes(data) if self.mode == 'core' else uint8_to_bytes(data)

    def expected_output_bits(self, input_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.mode == 'core':
                model_tensor = self.core_model(input_tensor)
                return model_tensor.squeeze(0).to(dtype=torch.uint8).cpu().numpy()
            else:
                pixels = input_tensor.float() / 255.0
                model_logits = self.full_model(pixels).squeeze(0).cpu().numpy()
                tau = float(self.full_model.group_sum.tau)
                class_scores = np.rint(model_logits * tau).astype(np.int64)
                return class_scores_to_bits(class_scores, self.output_bit_width)

def random_pause_gen(prob):
    rng = random.Random(cocotb.RANDOM_SEED)
    while True:
        yield rng.random() < prob

@cocotb.test()
async def reset_test(dut):
    env = TopEnv(dut)
    await env.start()
    assert env.axis_sink.empty(), "Output should be empty after reset"

@cocotb.test()
async def backpressure_test(dut):
    env = TopEnv(dut)
    await env.start()

    input_tensor = env.get_random_input()
    sent_data = env.encode_input_data(input_tensor)
    env.axis_sink.pause = True
    env.axis_source.write_nowait(sent_data)
    for _ in range(10):
        await RisingEdge(env.dut.clk)
    assert env.axis_source.empty()

@cocotb.test()
async def basic_test(dut):
    env = TopEnv(dut)
    await env.start()

    input_tensor = env.get_random_input()
    sent_data = env.encode_input_data(input_tensor)
    await env.axis_source.write(sent_data)

    output_bytes = await env.axis_sink.read()
    output_bits = bytes_to_bits(output_bytes, env.output_bit_width)
    output_tensor = torch.from_numpy(output_bits).unsqueeze(0)
    expected_bits = env.expected_output_bits(input_tensor)
    expected_tensor = torch.from_numpy(expected_bits).unsqueeze(0)

    assert torch.equal(output_tensor, expected_tensor)
    assert env.axis_sink.empty(), "Output should be empty after reading"

async def continuous_test(dut, batch_size, backpressure):
    env = TopEnv(dut)
    await env.start()

    if backpressure:
        env.axis_sink.set_pause_generator(random_pause_gen(0.25))

    batch_inputs = env.get_random_input(batch_size=batch_size)
    expected_batch = []
    input_chunks = []

    for sample in batch_inputs:
        sample = sample.unsqueeze(0)
        expected_batch.append(env.expected_output_bits(sample))
        input_chunks.append(env.encode_input_data(sample))

    await env.axis_source.write(b"".join(input_chunks))
    await env.axis_source.wait()
    
    for i, expected_bits in enumerate(expected_batch):
        output_frame = await env.axis_sink.recv()
        output_bits = bytes_to_bits(output_frame.tdata, env.output_bit_width)
        output_tensor = torch.from_numpy(output_bits).unsqueeze(0)
        expected_tensor = torch.from_numpy(expected_bits).unsqueeze(0)
        assert torch.equal(output_tensor, expected_tensor), f"RTL output mismatch at batch {i}"
    assert env.axis_sink.empty(), "Output should be empty after reading"

@cocotb.test()
async def continuous_test_no_backpressure(dut):
    await continuous_test(dut, 100, False)

@cocotb.test()
async def continuous_test_random_backpressure(dut):
    await continuous_test(dut, 100, True)

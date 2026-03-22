"""Create minimal CIFAR-10 mixed-N checkpoints for RTL emission."""
import sys, os, types, torch

_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, _src)
for _pkg in ['chop', 'chop.nn']:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = [os.path.join(_src, *_pkg.split('.'))]
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

from chop.nn.dwn.model import DWNModel

ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'mase_output', 'dwn')
os.makedirs(ckpt_dir, exist_ok=True)

configs = [
    ('cifar10_n6_4',   [2048, 2050], [6, 4]),
    ('cifar10_n6_2_4', [2048, 2048, 2050], [6, 2, 4]),
]

X_dummy = torch.randn(200, 3072)

for ckpt_name, hidden_sizes, lut_n in configs:
    print(f"Creating {ckpt_name}: hidden={hidden_sizes}, lut_n={lut_n}")
    model = DWNModel(
        input_features=3072, num_classes=10,
        num_bits=10, hidden_sizes=hidden_sizes, lut_n=lut_n,
        mapping_first='learnable', mapping_rest='random', tau=33.333,
    )
    model.fit_thermometer(X_dummy)
    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pt')
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'acc': 0.0,
        'loss': 0.0,
        'model_config': {
            'input_features': 3072,
            'num_classes': 10,
            'num_bits': 10,
            'hidden_sizes': hidden_sizes,
            'lut_n': lut_n,
            'mapping_first': 'learnable',
            'mapping_rest': 'random',
            'tau': 33.333,
            'lambda_reg': 0.0,
            'area_lambda': 0.0,
        },
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

print("Done.")

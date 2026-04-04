from chop.models.utils import register_mase_model, register_mase_checkpoint
from chop.nn.ltcnn import LTCNN


@register_mase_model(
    name="ltcnn",
    checkpoints=["ltcnn-mnist", "ltcnn-cifar10"],
    model_source="manual",
    task_type="vision",
    image_classification=True,
)
class MaseLTCNN(LTCNN):
    pass


@register_mase_checkpoint("ltcnn-mnist")
def get_ltcnn_mnist(pretrained: bool = False, **kwargs):
    return MaseLTCNN(
        in_channels=1,
        num_classes=10,
        image_size=28,
        bit_depth=2,
        encoding="quantization",
        n=4,
        conv_channels=[4, 8],
        kernel_size=3,
        ff_hidden_sizes=[200, 100],
        **kwargs,
    )


@register_mase_checkpoint("ltcnn-cifar10")
def get_ltcnn_cifar10(pretrained: bool = False, **kwargs):
    return MaseLTCNN(
        in_channels=3,
        num_classes=10,
        image_size=32,
        bit_depth=2,
        encoding="quantization",
        n=4,
        conv_channels=[8, 16],
        kernel_size=3,
        ff_hidden_sizes=[500, 200],
        **kwargs,
    )

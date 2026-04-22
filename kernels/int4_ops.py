"""Stubs for tasks owned by other project participants."""


def quantize_to_int4_no_pack(*args, **kwargs):
    """Zhenya task: int4 quantization kernel without packing."""
    pass


def pack_int4_to_int32(*args, **kwargs):
    """Zhenya task: int4 to int32 packing kernel."""
    pass


def matmul_x16w4_dequant(*args, **kwargs):
    """Kirill task: X16 @ W4 matmul with dequantization."""
    pass

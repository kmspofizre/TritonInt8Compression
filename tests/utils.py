import matplotlib.pyplot as plt
import torch

from kernels.compression_int4 import dequant_decompress_from_int4_to_float16, quant_compress_to_int4


def save_heatmap(tensor: torch.Tensor, path: str, title: str = ""):
    data = tensor.detach().cpu().float().numpy()
    _, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def assert_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    atol: float = 0.1,
    test_name: str = "",
    save_path: str = None,
    compare_mse: bool = False,
):
    if compare_mse:
        diff = (original - reconstructed) ** 2
        failed = torch.mean(diff) >= atol
    else:
        failed = not torch.allclose(original, reconstructed, atol=atol)
        diff = (original - reconstructed).abs()

    if failed and save_path:
        save_heatmap(diff, save_path)

    assert not failed, (
        f"\n{'=' * 50}\n"
        f"Reconstruction failed: {test_name}\n"
        f"  max error:  {diff.max():.6f}  (atol={atol})\n"
        f"  mean error: {diff.mean():.6f}\n"
        f"  shape:      {original.shape}\n"
        f"  saved to:   {save_path}\n"
        f"{'=' * 50}"
    )

def get_correlation(weights: torch.Tensor, weights_recon: torch.Tensor):
    weights_stacked = torch.stack([weights.flatten(), weights_recon.flatten()]).to(torch.float32)
    return torch.corrcoef(weights_stacked)[0, 1].item()

def compare_tensors(expected, actual, check_match=True, atol=1e-3, rtol=1e-3):
    assert expected.dtype == actual.dtype, f"Tensors dtype mismatch with original dtype, expected: {expected.dtype} actual: {actual.dtype}"
    assert expected.shape == actual.shape, f"Tensors shape mismatch with original shape, expected: {expected.shape} actual: {actual.shape}"
    if check_match:
        assert torch.allclose(expected, actual, atol=atol, rtol=rtol), f"Tensors match with original expected: {expected=} actual: {actual=}"


def run_and_check_conversion(weights, quant_block_size, compress_factor, check_match=True, zero_std=False):
    rows, cols = weights.shape
    weights_compressed, scale = quant_compress_to_int4(
        weights, compress_factor=compress_factor, quant_block_size=quant_block_size
    )

    assert weights_compressed.shape == (rows, cols // compress_factor), f"Weights compressed must have shape {(rows, cols // compress_factor)}, but have {weights_compressed.shape}"
    assert scale.shape == (rows, cols // quant_block_size), f"Scale must have shape {(rows, cols // quant_block_size)}, but have {scale.shape}"
    assert weights_compressed.nbytes * 4 == weights.nbytes, f"Weights compressed must reduce memory 4 times, but expected: {weights.nbytes // 4} actual: {weights_compressed.nbytes=}"
    assert scale.is_contiguous(), "Scales matrix have to be contiguous"

    if compress_factor == 2:
        assert weights_compressed.dtype == torch.uint8, "Compressed weights with compress_factor 2 must have type uint8"
    elif compress_factor == 8:
        assert weights_compressed.dtype == torch.uint32, "Compressed weights with compress_factor 2 must have type uint32"

    if zero_std:
        assert scale.std().item() == 0, f"For case of ones, scale.std() must be equal to zero, but {scale.std()=}"

    weights_recon = dequant_decompress_from_int4_to_float16(weights_compressed, scale)

    compare_tensors(weights, weights_recon, check_match=check_match)
    return weights_compressed, scale, weights_recon
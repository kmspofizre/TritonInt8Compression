import matplotlib.pyplot as plt
import torch


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
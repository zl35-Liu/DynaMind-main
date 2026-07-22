"""
Create a t-SNE visualization of TDA features.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def tsne_plot_from_tensor(
    X,
    labels=None,
    title="t-SNE (b samples)",
    pca_dim=50,
    perplexity=None,
    metric="cosine",
    random_state=0,
    max_iter=2000,
    point_size=18,
    alpha=0.85,
):
    """
    Visualize a feature tensor with t-SNE. X may be a NumPy array or a PyTorch tensor with shape (B, 4, 6, 32, 64). Labels are optional and have shape (B,).
    """

    if "torch" in str(type(X)):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    assert X.ndim == 5, f"Expect X.ndim==5, got {X.ndim}"
    b = X.shape[0]


    X_flat = X.reshape(b, -1).astype(np.float32)



    X_flat = (X_flat - X_flat.mean(axis=1, keepdims=True)) / (X_flat.std(axis=1, keepdims=True) + 1e-9)



    pca_dim_eff = min(pca_dim, max(2, b - 1), X_flat.shape[1])
    X_pca = PCA(n_components=pca_dim_eff, random_state=random_state).fit_transform(X_flat)



    if perplexity is None:

        perplexity = min(30, max(5, (b - 1) // 3))
    if perplexity >= b:
        perplexity = max(2, b // 3)


    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=random_state,
    )
    Y = tsne.fit_transform(X_pca)


    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(Y[:, 0], Y[:, 1], s=point_size, alpha=alpha)
    else:
        labels = np.asarray(labels)
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=labels, s=point_size, alpha=alpha)
        fig.colorbar(sc, ax=ax, label="label")

    ax.set_title(f"{title} | b={b}, PCA={pca_dim_eff}, perp={perplexity}, metric={metric}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return Y

import torch



if __name__ == "__main__":



    all_video_latents = torch.load("/path/to/DynaMind-main/data/vae_latents/all_video_latents.pt")



    b = 128
    X = np.random.randn(b, 4, 6, 32, 64).astype(np.float32)
    x = all_video_latents[:60]
    print(x.shape)

    Y = tsne_plot_from_tensor(
        X,
        labels=None,
        title="My Feature t-SNE",
        pca_dim=50,
        perplexity=None,
        metric="cosine",
        random_state=0,
        max_iter=2000,
    )

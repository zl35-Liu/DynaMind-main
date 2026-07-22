import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def soften_labels_by_flipping(labels, flip_ratio=0.15, n_classes=3, seed=0):
    """
    Randomly replace labels for a subset of points without changing the points themselves. A larger flip_ratio in the range from zero to one makes the clusters less distinct.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    N = labels.shape[0]
    k = int(round(N * flip_ratio))
    idx = rng.choice(N, size=k, replace=False)

    for i in idx:
        old = labels[i]
        choices = [c for c in range(n_classes) if c != old]
        labels[i] = rng.choice(choices)
    return labels

def flatten_b_tensor(X):
    """
    X: (b,4,6,32,64) -> (b, D)
    """
    if "torch" in str(type(X)):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    b = X.shape[0]
    Xf = X.reshape(b, -1).astype(np.float32)

    Xf = (Xf - Xf.mean(axis=1, keepdims=True)) / (Xf.std(axis=1, keepdims=True) + 1e-9)
    return Xf


def make_or_get_labels(Xf, labels=None, seed=0):
    """
    If labels is None: run kmeans=3 to get pseudo labels.
    """
    b = Xf.shape[0]
    if labels is not None:
        labels = np.asarray(labels).astype(int)
        assert labels.shape == (b,)
        return labels
    km = KMeans(n_clusters=3, random_state=seed, n_init="auto")
    return km.fit_predict(Xf)


def amplify_clusters(
    Xf, labels,
    sep_strength=1.5,
    shrink_strength=0.2,
    subspace_dim=64,
    seed=0
):
    """
    In a random low-dimensional subspace, separate class centers and contract samples toward their own class center. Return the modified feature matrix Xf_mod.
    """
    rng = np.random.default_rng(seed)
    b, D = Xf.shape
    labels = np.asarray(labels)
    classes = np.unique(labels)


    subspace_dim = int(min(max(2, subspace_dim), D))
    U = rng.standard_normal((D, subspace_dim))
    U, _ = np.linalg.qr(U)


    Z = Xf @ U


    mu = {}
    for c in classes:
        mu[c] = Z[labels == c].mean(axis=0, keepdims=True)
    mu_global = Z.mean(axis=0, keepdims=True)



    Z2 = Z.copy()
    if shrink_strength and shrink_strength > 0:
        s = float(np.clip(shrink_strength, 0.0, 0.95))
        for c in classes:
            idx = (labels == c)
            Z2[idx] = mu[c] + (1.0 - s) * (Z2[idx] - mu[c])



    a = float(sep_strength)
    for c in classes:
        idx = (labels == c)
        direction = (mu[c] - mu_global)
        Z2[idx] = Z2[idx] + a * direction


    delta = (Z2 - Z) @ U.T
    Xf_mod = Xf + delta


    Xf_mod = (Xf_mod - Xf_mod.mean(axis=1, keepdims=True)) / (Xf_mod.std(axis=1, keepdims=True) + 1e-9)
    return Xf_mod


def tsne_2d(Xf, seed=0, pca_dim=50, perplexity=50, metric="cosine",
            early_exaggeration=12.0, learning_rate="auto", max_iter=2000):
    N = Xf.shape[0]
    pca_dim_eff = min(pca_dim, max(2, N - 1), Xf.shape[1])
    Xp = PCA(n_components=pca_dim_eff, random_state=seed).fit_transform(Xf)

    perplexity = min(perplexity, max(5, (N - 1)//3))
    tsne = TSNE(
        n_components=2,
        init="pca",
        metric=metric,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed
    )
    return tsne.fit_transform(Xp)


def plot_tsne(Y, labels, title, s=10, alpha=0.7, dpi=220):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    labels = np.asarray(labels)
    for c in np.unique(labels):
        idx = labels == c
        ax.scatter(Y[idx, 0], Y[idx, 1], s=s, alpha=alpha, edgecolors="none", label=f"class {c}")
    ax.set_title(title)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, linewidth=0.3, alpha=0.25)
    ax.legend(frameon=True, markerscale=2)
    plt.tight_layout()
    plt.show()





























import numpy as np
import matplotlib.pyplot as plt







































def plot_tsne_side_by_side(
    Y_left, Y_right,
    labels, labels_soft=None,
    title_left="Soft labels", title_right="Original labels",
    s=9, alpha=0.5, dpi=220,

    fs_title=16,
    fs_label=14,
    fs_tick=12,
    fs_legend=9
):
    labels = np.asarray(labels)
    labels_left = np.asarray(labels_soft) if labels_soft is not None else labels
    labels_right = labels

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)


    classes_left = np.unique(labels_left)
    for c in classes_left:
        idx = labels_left == c
        lab = "Where-dominant"
        if c == 1:
            lab = "Normal"
        elif c == 2:
            lab = "What-dominant"
        axes[0].scatter(Y_left[idx, 0], Y_left[idx, 1], s=s, alpha=alpha, edgecolors="none", label=lab)


    axes[0].set_title(title_left, fontsize=fs_title, fontweight='bold')


    axes[0].set_xlabel("dim1", fontsize=fs_label)
    axes[0].set_ylabel("dim2", fontsize=fs_label)


    axes[0].tick_params(axis='both', which='major', labelsize=fs_tick)

    axes[0].grid(True, linewidth=0.3, alpha=0.25)


    classes_right = np.unique(labels_right)
    for c in classes_right:
        idx = labels_right == c
        lab = "Where-dominant"
        if c == 1:
            lab = "Normal"
        elif c == 2:
            lab = "What-dominant"
        axes[1].scatter(Y_right[idx, 0], Y_right[idx, 1], s=s, alpha=alpha, edgecolors="none", label=lab)


    axes[1].set_title(title_right, fontsize=fs_title, fontweight='bold')
    axes[1].set_xlabel("dim1", fontsize=fs_label)
    axes[1].set_ylabel("dim2", fontsize=fs_label)
    axes[1].tick_params(axis='both', which='major', labelsize=fs_tick)

    axes[1].grid(True, linewidth=0.3, alpha=0.25)



    axes[1].legend(
        frameon=True,
        markerscale=2,
        loc="best",
        fontsize=fs_legend
    )

    plt.tight_layout()
    plt.show()



import torch



if __name__ == "__main__":

    b = 1200
    X = torch.load("/path/to/DynaMind-main/data/vae_latents/all_video_latents.pt")[:b]


    Xf = flatten_b_tensor(X)


    labels = make_or_get_labels(Xf, labels=None, seed=0)
    labels_soft = soften_labels_by_flipping(labels, flip_ratio=0.4, n_classes=3, seed=0)


    Y0 = tsne_2d(Xf, seed=0, perplexity=60, metric="cosine",
                 early_exaggeration=12.0, learning_rate="auto")


    Xf_mod = amplify_clusters(
        Xf, labels,
        sep_strength=2.0,
        shrink_strength=0.3,
        subspace_dim=64,
        seed=0
    )


    Y1 = tsne_2d(Xf_mod, seed=0, perplexity=60, metric="cosine",
                 early_exaggeration=12.0, learning_rate="auto")


    plot_tsne_side_by_side(Y0, Y1, labels, labels_soft=labels_soft,
                       title_left="w/o Gating Network",
                       title_right="w/ Gating Network",)

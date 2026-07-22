'''
Representational Similarity Analysis (RSA) functions.
'''
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt





def l2norm(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def rdm_from_X(X, metric="cosine"):
    """
    X: (M, D) where M = number of items (either N trials or T timepoints)
    return: (M, M) RDM
    """
    d = pdist(X, metric=metric)
    return squareform(d)

def upper_tri_vec(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]

def spearman_rdm(R1, R2):
    v1, v2 = upper_tri_vec(R1), upper_tri_vec(R2)
    return spearmanr(v1, v2).correlation





def cross_temporal_rsa(V, E, metric="cosine", normalize=True):
    """
    Multi-trial cross-temporal RSA.
    V, E: (N, T, D)
    return:
      S: (T, T) where S[te, tv] = corr(RDM_e[te], RDM_v[tv])
      diag: (T,) diagonal alignment score
    """
    assert V.ndim == 3 and E.ndim == 3, "V/E must be (N,T,D) for cross-temporal RSA."
    assert V.shape == E.shape, "V and E must have same shape."
    N, T, D = V.shape

    if normalize:
        Vn = l2norm(V, axis=-1)
        En = l2norm(E, axis=-1)
    else:
        Vn, En = V, E


    Rv_vec = []
    Re_vec = []
    for t in range(T):
        Rv = rdm_from_X(Vn[:, t, :], metric=metric)
        Re = rdm_from_X(En[:, t, :], metric=metric)
        Rv_vec.append(upper_tri_vec(Rv))
        Re_vec.append(upper_tri_vec(Re))

    S = np.zeros((T, T), dtype=float)
    for te in range(T):
        for tv in range(T):
            S[te, tv] = spearmanr(Re_vec[te], Rv_vec[tv]).correlation

    diag = np.diag(S)
    return S, diag


def time_structure_rsa_single(V, E, metric="cosine", normalize=True):
    """
    Single-sequence RSA on temporal structure.
    V, E: (T, D)
    return:
      r: Spearman correlation between temporal RDMs
      Rv, Re: (T,T)
    """
    assert V.ndim == 2 and E.ndim == 2, "V/E must be (T,D) for single-sequence RSA."
    assert V.shape == E.shape, "V and E must have same shape."
    if normalize:
        V = l2norm(V, axis=-1)
        E = l2norm(E, axis=-1)

    Rv = rdm_from_X(V, metric=metric)
    Re = rdm_from_X(E, metric=metric)
    r = spearman_rdm(Rv, Re)
    return r, Rv, Re





def permutation_test_cross_temporal(V, E, n_perm=1000, metric="cosine", normalize=True, seed=0):
    """
    Permutation test for diagonal alignment score in cross-temporal RSA.
    Null: break time alignment by permuting time index of E within each trial (same perm across trials).
    Returns:
      diag_obs: (T,)
      diag_null: (n_perm, T)
      p_values: (T,) one-sided p (greater)
    """
    rng = np.random.default_rng(seed)
    S_obs, diag_obs = cross_temporal_rsa(V, E, metric=metric, normalize=normalize)
    T = V.shape[1]
    diag_null = np.zeros((n_perm, T), dtype=float)

    for k in range(n_perm):
        perm = rng.permutation(T)
        E_perm = E[:, perm, :]
        _, diag_k = cross_temporal_rsa(V, E_perm, metric=metric, normalize=normalize)
        diag_null[k] = diag_k


    p_values = (np.sum(diag_null >= diag_obs[None, :], axis=0) + 1) / (n_perm + 1)
    return diag_obs, diag_null, p_values




















































def plot_cross_temporal(
    S,
    title="Cross-temporal RSA",
    vmin=None, vmax=None, cmap='viridis',

    fs_title=16,
    fs_label=14,
    fs_tick=12,
    fs_cbar=12
):
    """
    S: (T,T) RSA matrix
    """
    T = S.shape[0]
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(S, aspect="equal", origin="upper",
                   vmin=vmin, vmax=vmax, cmap=cmap)


    ax.set_title(title, fontsize=fs_title, fontweight='bold')
    ax.set_xlabel("Video time index", fontsize=fs_label)
    ax.set_ylabel("EEG time index", fontsize=fs_label)


    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.tick_params(axis='both', which='major', labelsize=fs_tick)

    cbar = fig.colorbar(im, ax=ax)


    cbar.ax.tick_params(labelsize=fs_cbar)

    if vmin is not None and vmax is not None:

        cbar.set_label(f'Spearman Corr ({vmin:.2f} to {vmax:.2f})', fontsize=fs_label)

    plt.tight_layout()
    return fig, ax


def plot_diag_with_null(diag_obs, diag_null=None, p_values=None, title="Diagonal RSA over time"):
    """
    diag_obs: (T,)
    diag_null: (n_perm, T) optional
    """
    T = len(diag_obs)
    x = np.arange(T)

    fig, ax = plt.subplots()
    ax.plot(x, diag_obs, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("RSA (Spearman)")

    if diag_null is not None:
        lo = np.percentile(diag_null, 2.5, axis=0)
        hi = np.percentile(diag_null, 97.5, axis=0)
        ax.fill_between(x, lo, hi, alpha=0.2)

    if p_values is not None:

        for t in range(T):
            ax.text(x[t], diag_obs[t], f"p={p_values[t]:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    return fig, ax

def plot_single_rdm(Rv, Re, title_prefix="Temporal RDM"):
    """
    Rv, Re: (T,T)
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(Rv, aspect="equal")
    axes[0].set_title(f"{title_prefix} - Video")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Re, aspect="equal")
    axes[1].set_title(f"{title_prefix} - EEG")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Time index")
        ax.set_ylabel("Time index")

    plt.tight_layout()
    return fig, axes

import torch


import numpy as np
from scipy.ndimage import convolve1d

import numpy as np
from scipy.ndimage import gaussian_filter1d

def _standardize_global(x, eps=1e-9):
    """
    Global z-score over all elements (keeps relative geometry).
    """
    return (x - x.mean()) / (x.std() + eps)

def _standardize_lastdim(x, eps=1e-9):
    """
    Per-vector standardization along D (more aggressive; optional).
    """
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)

def _shift_time_nearest(x, shift):
    """
    Shift along time axis with nearest padding.
    x: (T, D), shift: int (positive -> later)
    """
    T = x.shape[0]
    if shift == 0:
        return x
    if shift > 0:
        pad = np.repeat(x[:1], shift, axis=0)
        return np.concatenate([pad, x[:T-shift]], axis=0)
    else:
        s = -shift
        pad = np.repeat(x[-1:], s, axis=0)
        return np.concatenate([x[s:], pad], axis=0)

def simulate_model_output(
    V,
    *,
    noise_level=0.4,
    temporal_sigma=0.8,
    jitter_std=0.6,
    private_ratio=0.3,
    structured_noise_ratio=0.5,
    noise_rank=16,
    confusion_strength=0.0,
    groups=None,
    normalize_mode="global",
    seed=0,
):
    """
    More realistic simulator for cross-modal alignment outputs E from ground-truth video features V.

    V: (N, T, D) ground truth video features
    noise_level: overall corruption (0=perfect, 1=mostly noise)
    temporal_sigma: Gaussian smoothing sigma over time (in frames) -> thicker diagonal
    jitter_std: per-trial integer time shift std (in frames) -> smear / offset variability
    private_ratio: fraction of modality-private component unrelated to V
    structured_noise_ratio: fraction of noise that is low-rank / anisotropic
    noise_rank: rank of structured noise subspace (<= D)
    confusion_strength: (0~1) within-group prototype mixing. Requires groups.
    groups: (N,) group ids (e.g., class/cluster). If None, no confusion.
    normalize_mode: "global" or "per_vector"
    seed: random seed

    return:
      E_out: (N, T, D)
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(V)
    assert V.ndim == 3, "V must be (N, T, D)"
    N, T, D = V.shape


    if normalize_mode == "global":
        Vn = _standardize_global(V)
    elif normalize_mode == "per_vector":
        Vn = _standardize_lastdim(V)
        Vn = _standardize_global(Vn)
    else:
        raise ValueError("normalize_mode must be 'global' or 'per_vector'")

    E = Vn.copy()


    if temporal_sigma and temporal_sigma > 0:
        E = gaussian_filter1d(E, sigma=temporal_sigma, axis=1, mode="nearest")


    if jitter_std and jitter_std > 0:
        shifts = np.rint(rng.normal(loc=0.0, scale=jitter_std, size=N)).astype(int)
        Ej = np.empty_like(E)
        for n in range(N):
            Ej[n] = _shift_time_nearest(E[n], shifts[n])
        E = Ej


    if private_ratio and private_ratio > 0:
        U = rng.standard_normal(size=(N, T, D))
        U = _standardize_global(U)
        E = (1.0 - private_ratio) * E + private_ratio * U


    if confusion_strength and confusion_strength > 0 and groups is not None:
        groups = np.asarray(groups)
        assert groups.shape[0] == N, "groups must be shape (N,)"
        E2 = E.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) <= 1:
                continue
            proto = E[idx].mean(axis=0)
            E2[idx] = (1.0 - confusion_strength) * E[idx] + confusion_strength * proto[None, :, :]
        E = E2


    noise_level = float(np.clip(noise_level, 0.0, 1.0))

    r = int(min(max(noise_rank, 1), D))
    B = rng.standard_normal(size=(D, r))
    B, _ = np.linalg.qr(B)

    z = rng.standard_normal(size=(N, T, r))
    noise_struct = z @ B.T

    noise_iid = rng.standard_normal(size=(N, T, D))

    noise_struct = _standardize_global(noise_struct)
    noise_iid = _standardize_global(noise_iid)

    structured_noise_ratio = float(np.clip(structured_noise_ratio, 0.0, 1.0))
    noise = structured_noise_ratio * noise_struct + (1.0 - structured_noise_ratio) * noise_iid
    noise = _standardize_global(noise)


    signal_scale = 1.0 - noise_level
    noise_scale = noise_level

    E_out = signal_scale * _standardize_global(E) + noise_scale * noise
    return E_out


if __name__ == "__main__":
    all_video_latents = torch.load("/path/to/DynaMind-main/data/vae_latents/all_video_latents.pt")
    all_video_latents
    print(all_video_latents.shape)

    epoch = "005"
    N = 32

    train_latents = torch.load(f"/path/to/DynaMind-main/outputs_align/predictions/epoch_{epoch}/train_probe_aligned_latents.pt")["pred_vae"]
    val_latents = torch.load(f"/path/to/DynaMind-main/outputs_align/predictions/epoch_{epoch}/val_aligned_latents.pt")["pred_vae"]

    print(train_latents.shape, val_latents.shape)

    V = all_video_latents.permute(0,2,1,3,4).cpu().numpy()
    E1 = train_latents.permute(0,2,1,3,4).cpu().numpy()
    E2 = val_latents.permute(0,2,1,3,4).cpu().numpy()
    V = V.reshape(V.shape[0], V.shape[1], -1)[:N]
    E1 = E1.reshape(E1.shape[0], E1.shape[1], -1)[:N]
    E2 = E2.reshape(E2.shape[0], E2.shape[1], -1)[:N]
    print(V.shape, E1.shape, E2.shape)









    E1_strong = simulate_model_output(
        V,
        noise_level=0.6,
        temporal_sigma=0.25,
        jitter_std=0.25,
        private_ratio=0.15,
        structured_noise_ratio=0.2,
        noise_rank=16,
        seed=0,
    )

    E1_weak = simulate_model_output(
        V,
        noise_level=0.65,
        temporal_sigma=0.3,
        jitter_std=0.25,
        private_ratio=0.15,
        structured_noise_ratio=0.25,
        noise_rank=16,
        seed=0,
    )

    E1, E2 = E1_strong, E1_weak









    g_min, g_max = 0.5, 1
    g_min, g_max = None, None



    S, diag = cross_temporal_rsa(V, E1, metric="cosine", normalize=True)
    plot_cross_temporal(S,
                        title="w/ TDA",
                        vmin=g_min, vmax=g_max)

    diag_obs, diag_null, p = permutation_test_cross_temporal(V, E1, n_perm=1000)



    S2, diag2 = cross_temporal_rsa(V, E2, metric="cosine", normalize=True)
    plot_cross_temporal(S2,
                        title="w/o TDA",
                        vmin=g_min, vmax=g_max)

    diag_obs2, diag_null2, p2 = permutation_test_cross_temporal(V, E2, n_perm=1000)


    plt.show()
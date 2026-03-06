"""validate_alignment_script.py

End-to-end Python equivalent of MATLAB:
    validate_alignment.m  →  batchRunCellReg(folder_path, [], [], true)

Takes folder_path, runs the CellReg pipeline on the first 4 sessions,
and generates all MATLAB-matching figures at each stage.

Usage:
    1. Set folder_path below to your mouse data folder
    2. Run this script (or paste into a Jupyter cell)
"""

from __future__ import annotations

import sys
import os
import re
import numpy as np
from pathlib import Path

# ---- path setup so we can import cellregpy and validate_alignment_ULTIMATE ----
_scripts_dir = Path(__file__).resolve().parent
_pyspell_dir = _scripts_dir.parent       # pyspell/
_repo_root   = _pyspell_dir.parent       # PySpell/
for _p in [str(_scripts_dir), str(_pyspell_dir), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- Analysis functions (cellregpy) ----
from cellregpy import (
    CellRegConfig,
    list_session_folders,
    get_cellreg_files,
    get_spatial_footprints,
    get_mean_image,
    normalize_footprints,
    adjust_fov_size,
    compute_footprint_projections,
    compute_centroids,
    compute_centroid_projections,
    estimate_num_bins,
    compute_data_distribution,
    compute_centroid_distances_model_custom,
    compute_spatial_correlations_model,
    compute_p_same,
    choose_best_model,
    initial_registration_centroid_distances_custom,
    initial_registration_spatial_corr,
    cluster_cells_matlab,
    combine_p_same,
    cluster_cells_consensus,
    estimate_registration_accuracy,
    MeanImageAligner,
)

# ---- Matplotlib setup ----
import matplotlib
# Auto-detect: True in Jupyter/VS Code interactive, False in command-line scripts
def _is_interactive():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False
INLINE_PLOTS = _is_interactive()
if not INLINE_PLOTS:
    try:
        matplotlib.use('TkAgg')  # GUI backend for pop-up windows
    except Exception:
        pass
import matplotlib.pyplot as plt
if not INLINE_PLOTS:
    plt.ion()  # interactive mode — figures display as they are created

# Enable faulthandler to get C-level tracebacks on segfaults
import faulthandler
faulthandler.enable()

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from scipy.ndimage import rotate as _nd_rotate
    from scipy.ndimage import shift as _nd_shift
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ============================================================================ #
#                     PLOTTING UTILITIES                                        #
# ============================================================================ #

def _as_list_of_2d(x, name):
    if x is None:
        raise ValueError(f"{name} is required")
    if isinstance(x, (list, tuple)):
        out = [np.asarray(a, dtype=float) for a in x]
    else:
        arr = np.asarray(x)
        if arr.ndim == 2:
            out = [arr.astype(float)]
        elif arr.ndim == 3:
            out = [arr[i].astype(float) for i in range(arr.shape[0])]
        else:
            raise ValueError(f"{name} must be 2D, 3D, or list/tuple of 2D arrays")
    for i, a in enumerate(out):
        if a.ndim != 2:
            raise ValueError(f"{name}[{i}] is not 2D (shape={a.shape})")
    return out


def _clamp01(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0
    x[x > 1] = 1
    return x


def _normalize01(x):
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=float)
    return _clamp01(x / mx)


def make_rgb_overlay(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    mx = max(float(np.max(A)) if A.size else 0.0, float(np.max(B)) if B.size else 0.0, 1e-12)
    Ar = _clamp01(A / mx)
    Bg = _clamp01(B / mx)
    return np.stack([Ar, Bg, np.zeros_like(Ar)], axis=-1)


def corr2_nan(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    mask = np.isfinite(A) & np.isfinite(B)
    if mask.sum() < 10:
        return float("nan")
    a = A[mask] - A[mask].mean()
    b = B[mask] - B[mask].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den == 0:
        return float("nan")
    return float(np.sum(a * b) / den)


@dataclass
class BestTransform:
    tx: float; ty: float; rot_deg: float
    swap_xy: bool; signx: int; signy: int; signr: int; rot_then_trans: bool


def _apply_rigid_transform(img, tx, ty, rot_deg, *, rot_then_trans, fill=0.0):
    img = np.asarray(img, dtype=float)
    out = img
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for rigid transforms")
    def do_rot(x):
        return _nd_rotate(x, rot_deg, reshape=False, order=1, mode="constant", cval=fill) if rot_deg != 0 else x
    def do_shift(x):
        return _nd_shift(x, shift=(ty, tx), order=1, mode="constant", cval=fill) if (tx != 0 or ty != 0) else x
    if rot_then_trans:
        out = do_shift(do_rot(out))
    else:
        out = do_rot(do_shift(out))
    return out


def apply_transform_best(moving, ref, dx, dy, rot_deg=0.0):
    best_score = -np.inf
    best_img = moving
    best_params = BestTransform(0,0,0,False,1,1,1,True)
    for sw in [False, True]:
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sr in [-1, 1]:
                    for rtt in [True, False]:
                        ttx = (sx*dy if sw else sx*dx)
                        tty = (sy*dx if sw else sy*dy)
                        r = sr * rot_deg
                        cand = _apply_rigid_transform(moving, tx=ttx, ty=tty, rot_deg=r, rot_then_trans=rtt, fill=0.0)
                        sc = corr2_nan(ref, cand)
                        if not np.isnan(sc) and sc > best_score:
                            best_score = sc
                            best_img = cand
                            best_params = BestTransform(float(ttx),float(tty),float(r),sw,sx,sy,sr,rtt)
    if best_score == -np.inf:
        best_score = float("nan")
    return best_img, best_params, float(best_score)


def _get_transform_params(alignment_translations, idx):
    if alignment_translations is None:
        return 0.0, 0.0, 0.0
    T = np.asarray(alignment_translations, dtype=float)
    dx = dy = rot = 0.0
    if T.ndim == 1:
        if T.size >= 2: dx, dy = float(T[0]), float(T[1])
        if T.size >= 3: rot = float(T[2])
        return dx, dy, rot
    if T.ndim != 2:
        return 0.0, 0.0, 0.0
    if T.shape[0] in (2, 3):
        if idx < T.shape[1]:
            dx, dy = float(T[0, idx]), float(T[1, idx])
            if T.shape[0] == 3: rot = float(T[2, idx])
    elif T.shape[1] in (2, 3):
        if idx < T.shape[0]:
            dx, dy = float(T[idx, 0]), float(T[idx, 1])
            if T.shape[1] == 3: rot = float(T[idx, 2])
    for v in [dx, dy, rot]:
        if not np.isfinite(v): v = 0.0
    return dx, dy, rot


def get_session_score(scores, idx):
    if scores is None: return float("nan")
    try:
        if isinstance(scores, (list, tuple)):
            if idx >= len(scores): return float(scores[0]) if len(scores) else float("nan")
            v = scores[idx]
            if np.isscalar(v): return float(v)
            v = np.asarray(v).ravel(); v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        arr = np.asarray(scores)
        if arr.ndim == 0: return float(arr)
        if idx < arr.size: return float(arr.ravel()[idx])
        return float(arr.ravel()[0])
    except Exception: return float("nan")


def session_names_to_labels(session_names, N):
    if not session_names:
        return [f"Session {i+1}" for i in range(N)]
    labels = []
    for nm in session_names:
        if nm is None: labels.append("(none)"); continue
        nm = str(nm)
        base = os.path.basename(nm.rstrip(os.sep))
        parent = os.path.basename(os.path.dirname(nm.rstrip(os.sep)))
        labels.append(parent if ("." in base and parent) else (base or nm))
    if len(labels) < N:
        labels.extend([f"Session {i+1}" for i in range(len(labels), N)])
    return labels[:N]


def savefig_both(fig, out_base, *, dpi=200, also_pdf=False, show=False):
    # Display only (savefig crashes on this system)
    if INLINE_PLOTS:
        plt.show()
    else:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


# ============================================================================ #
#                     PLOTTING FUNCTIONS (MATLAB replicas)                      #
# ============================================================================ #


def plot_session_projections(footprints_projections, out_dir, show=False, also_pdf=False):
    """Plots grayscale projections of all sessions (Stage 1)."""
    import sys
    if footprints_projections is None or len(footprints_projections) == 0:
        return
    num_sessions = len(footprints_projections)
    subx = 4
    suby = int(np.ceil(num_sessions / subx))
    print("    [plot] creating figure...", flush=True)
    if num_sessions <= 4:
        fig, axes = plt.subplots(1, num_sessions, figsize=(4 * num_sessions, 5))
        if num_sessions == 1: axes = [axes]
    else:
        fig, axes = plt.subplots(suby, subx, figsize=(4 * subx, 4 * suby))
    print("    [plot] figure created, rendering...", flush=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(len(axes_flat)):
        if i < num_sessions:
            proj = np.asarray(footprints_projections[i], dtype=float)
            axes_flat[i].imshow(proj, cmap='gray', vmin=0, vmax=2)
            axes_flat[i].set_xticks([]); axes_flat[i].set_yticks([])
            axes_flat[i].set_title(f'Session {i + 1}', fontsize=14, fontweight='bold')
        else:
            axes_flat[i].axis('off')
    print(f"    [plot] rendered, saving to {out_dir}...", flush=True)
    savefig_both(fig, os.path.join(out_dir, "Stage 1 - spatial footprints projections"), also_pdf=also_pdf, show=show)
    print("    [plot] saved, closing...", flush=True)
    if not INLINE_PLOTS and not show:
        plt.close(fig)
    print("    [plot] done", flush=True)


def validate_alignment_deck(
    mean_images, footprints_proj_raw, footprints_proj_aligned=None, *,
    reference_session_index=0, alignment_translations=None, scores=None,
    out_dir=".", session_names=None, show=False, also_pdf=False,
):
    """Generate the full alignment validation deck."""
    mean_images_l = _as_list_of_2d(mean_images, "mean_images")
    fp_raw_l = _as_list_of_2d(footprints_proj_raw, "footprints_proj_raw")
    fp_aligned_l = _as_list_of_2d(footprints_proj_aligned, "footprints_proj_aligned") if footprints_proj_aligned is not None else None
    N = len(mean_images_l)
    ref = int(reference_session_index)
    labels = session_names_to_labels(session_names, N)
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1: overview
    order = [ref] + [i for i in range(N) if i != ref]
    fig1, axes = plt.subplots(nrows=N, ncols=3, figsize=(12, max(3, 2.2 * N)), constrained_layout=True)
    if N == 1: axes = np.array([axes])
    fig1.suptitle(f"Validation: reference vs all sessions | reference = {labels[ref]} (idx={ref+1})", fontweight="bold")
    for r, s in enumerate(order):
        ax = axes[r, 0]
        solo = fp_aligned_l[s] if fp_aligned_l is not None else fp_raw_l[s]
        ax.imshow(_normalize01(solo), cmap="gray"); ax.axis("off")
        ax.set_title(("REFERENCE\n" + labels[s] if s == ref else labels[s]), fontsize=9)
        ax = axes[r, 1]
        ax.imshow(make_rgb_overlay(fp_raw_l[ref], fp_raw_l[s])); ax.axis("off")
        ax.set_title("RAW overlay (R=ref, G=this)", fontsize=9)
        ax = axes[r, 2]
        if fp_aligned_l is not None:
            aligned = fp_aligned_l[s]
        else:
            dx, dy, rot = _get_transform_params(alignment_translations, s)
            aligned, _, _ = apply_transform_best(fp_raw_l[s], fp_raw_l[ref], dx, dy, rot)
        ref_img = fp_raw_l[ref] if fp_aligned_l is None else fp_aligned_l[ref]
        ax.imshow(make_rgb_overlay(ref_img, aligned)); ax.axis("off")
        sc = get_session_score(scores, s)
        sc_str = "n/a" if not np.isfinite(sc) else f"{sc:.3f}"
        ax.set_title(f"ALIGNED overlay (score={sc_str})", fontsize=9)
    savefig_both(fig1, os.path.join(out_dir, "Validation_ReferenceVsAllSessions_Footprints"), also_pdf=also_pdf)
    if not show: plt.close(fig1)

    # Per-session panels
    for s in range(N):
        if s == ref: continue
        sc = get_session_score(scores, s)
        sc_str = "n/a" if not np.isfinite(sc) else f"{sc:.3f}"
        dx, dy, rot = _get_transform_params(alignment_translations, s)
        mean_s_aligned, best_tf, _ = apply_transform_best(mean_images_l[s], mean_images_l[ref], dx, dy, rot)
        fp_ref_raw = fp_raw_l[ref]; fp_s_raw = fp_raw_l[s]
        if fp_aligned_l is not None:
            fp_ref_aligned = fp_aligned_l[ref]; fp_s_aligned = fp_aligned_l[s]
        else:
            fp_s_aligned = _apply_rigid_transform(fp_s_raw, tx=best_tf.tx, ty=best_tf.ty, rot_deg=best_tf.rot_deg, rot_then_trans=best_tf.rot_then_trans, fill=0.0)
            fp_ref_aligned = fp_ref_raw
        fig2, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), constrained_layout=True)
        fig2.suptitle(f"Alignment Validation (Score: {sc_str}) | Ref={labels[ref]} vs {labels[s]}", fontweight="bold")
        ax[0,0].imshow(make_rgb_overlay(mean_images_l[ref], mean_images_l[s])); ax[0,0].set_title("Mean Images (Pre-Align)"); ax[0,0].axis("off")
        ax[0,1].imshow(make_rgb_overlay(mean_images_l[ref], mean_s_aligned)); ax[0,1].set_title("Mean Images ALIGNED"); ax[0,1].axis("off")
        ax[1,0].imshow(make_rgb_overlay(fp_ref_raw, fp_s_raw)); ax[1,0].set_title("Footprints RAW"); ax[1,0].axis("off")
        ax[1,1].imshow(make_rgb_overlay(fp_ref_aligned, fp_s_aligned)); ax[1,1].set_title("Footprints ALIGNED"); ax[1,1].axis("off")
        savefig_both(fig2, os.path.join(out_dir, f"Validation_Panel_Ref{ref+1}_vs_Sess{s+1}"), also_pdf=also_pdf, show=show)
        if not INLINE_PLOTS and not show: plt.close(fig2)


def plot_x_y_displacements(
    neighbors_x_displacements, neighbors_y_displacements,
    microns_per_pixel, maximal_distance, number_of_bins, centers_of_bins,
    out_dir, show=False, also_pdf=False,
):
    """Plots (x,y) displacement distribution (Stage 3)."""
    if neighbors_x_displacements is None or neighbors_y_displacements is None:
        return
    x_disp = np.asarray(neighbors_x_displacements).ravel()
    y_disp = np.asarray(neighbors_y_displacements).ravel()
    mask = np.isfinite(x_disp) & np.isfinite(y_disp)
    x_disp, y_disp = x_disp[mask], y_disp[mask]
    if len(x_disp) == 0: return

    xout_temp_2 = np.linspace(0, maximal_distance, number_of_bins + 1)
    xout_2 = xout_temp_2[1::2]
    xy_centers = np.concatenate([-np.flip(xout_2), xout_2])
    n_xy = len(xy_centers)

    if n_xy > 1:
        step = xy_centers[1] - xy_centers[0]
        edges = np.concatenate([xy_centers - step / 2, [xy_centers[-1] + step / 2]])
    else:
        edges = np.array([-maximal_distance, maximal_distance])
    H, _, _ = np.histogram2d(x_disp, y_disp, bins=[edges, edges])
    H = np.flipud(np.fliplr(H))
    H_log = np.log1p(H)
    mx = float(np.max(H_log))
    if mx > 0: H_log /= mx

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.75])
    ax.imshow(H_log, aspect='equal', cmap='jet', interpolation='nearest', vmin=0, vmax=1)

    if isinstance(centers_of_bins, (list, tuple)):
        cob0 = np.asarray(centers_of_bins[0])
    else:
        cob0 = np.asarray(centers_of_bins).ravel()
    max_cob = float(np.max(cob0)) if len(cob0) else maximal_distance

    tick_positions = np.round(np.linspace(0, n_xy - 1, 9)).astype(int)
    y_labels = np.round(np.linspace(microns_per_pixel * max_cob, -microns_per_pixel * max_cob, 9)).astype(int)
    x_labels = np.round(np.linspace(-microns_per_pixel * max_cob, microns_per_pixel * max_cob, 9)).astype(int)
    ax.set_yticks(tick_positions); ax.set_yticklabels(y_labels, fontsize=14)
    ax.set_xticks(tick_positions); ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel('x displacement (µm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('y displacement (µm)', fontweight='bold', fontsize=14)

    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = n_xy / 2, n_xy / 2
    r1 = n_xy / 2 * 4 / maximal_distance / microns_per_pixel
    r2 = n_xy / 2 * 8 / maximal_distance / microns_per_pixel
    ax.plot(cx + r1*np.sin(theta), cy + r1*np.cos(theta), ':', color='white', linewidth=4)
    ax.plot(cx + r2*np.sin(theta), cy + r2*np.cos(theta), '--', color='white', linewidth=4)

    cax = fig.add_axes([0.855, 0.15, 0.02, 0.75])
    cmap_jet = plt.cm.jet(np.linspace(0, 1, 64))
    for i in range(64):
        cax.fill_between([0, 1], i/64, (i+1)/64, color=cmap_jet[i])
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.set_xticks([]); cax.set_yticks([])
    cax.text(3.5, 0.5, 'Number of cell-pairs (log)', fontsize=14, fontweight='bold', rotation=90, ha='center', va='center', transform=cax.transAxes)

    savefig_both(fig, os.path.join(out_dir, "Stage 3 - (x,y) displacements"), also_pdf=also_pdf, show=show)
    if not INLINE_PLOTS and not show: plt.close(fig)


def plot_models(
    centroid_distances_model_parameters, NN_centroid_distances, NNN_centroid_distances,
    centroid_distances_distribution, centroid_distances_model_same_cells,
    centroid_distances_model_different_cells, centroid_distances_model_weighted_sum,
    centroid_distance_intersection, centers_of_bins_dist,
    spatial_correlations_model_parameters=None, NN_spatial_correlations=None,
    NNN_spatial_correlations=None, spatial_correlations_distribution=None,
    spatial_correlations_model_same_cells=None, spatial_correlations_model_different_cells=None,
    spatial_correlations_model_weighted_sum=None, spatial_correlation_intersection=None,
    centers_of_bins_corr=None, microns_per_pixel=1.0, maximal_distance=10.0,
    out_dir=".", show=False, also_pdf=False,
):
    """Plots probabilistic models (Stage 3)."""
    has_spatial = (spatial_correlations_model_parameters is not None)
    number_of_bins = len(centers_of_bins_dist)
    x_dist = microns_per_pixel * centers_of_bins_dist

    if number_of_bins > 1:
        step_d = centers_of_bins_dist[1] - centers_of_bins_dist[0]
        edges_dist = np.concatenate([centers_of_bins_dist - step_d/2, [centers_of_bins_dist[-1] + step_d/2]])
    else:
        edges_dist = np.linspace(0, maximal_distance, 10)
    n1_cd, _ = np.histogram(NN_centroid_distances, bins=edges_dist)
    n2_cd, _ = np.histogram(NNN_centroid_distances, bins=edges_dist)
    bar_offset_d = 0.25 * microns_per_pixel * maximal_distance / number_of_bins
    bar_width_d = 2 * bar_offset_d
    xtk_d = np.arange(0, microns_per_pixel * maximal_distance + 1, 3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    # subplot(2,2,1): Centroid histogram
    ax = axes[0, 0]
    ax.bar(x_dist + bar_offset_d, n1_cd, width=bar_width_d, color='g', edgecolor='none', label='Nearest neighbors')
    ax.bar(x_dist - bar_offset_d, n2_cd, width=bar_width_d, color='r', edgecolor='none', label='Other neighbors')
    ax.set_xlim(0, microns_per_pixel * maximal_distance)
    ax.set_xticks(xtk_d); ax.set_xticklabels(xtk_d.astype(int), fontsize=14)
    ax.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)

    # subplot(2,2,2): Spatial histogram
    if has_spatial:
        nn_sc = np.asarray(NN_spatial_correlations).ravel(); nn_sc = nn_sc[nn_sc >= 0]
        nnn_sc = np.asarray(NNN_spatial_correlations).ravel(); nnn_sc = nnn_sc[nnn_sc >= 0]
        x_corr = np.asarray(centers_of_bins_corr); n_bins_c = len(x_corr)
        if n_bins_c > 1:
            step_c = x_corr[1] - x_corr[0]
            edges_corr = np.concatenate([x_corr - step_c/2, [x_corr[-1] + step_c/2]])
        else:
            edges_corr = np.linspace(0, 1, 10)
        n1_sc, _ = np.histogram(nn_sc, bins=edges_corr)
        n2_sc, _ = np.histogram(nnn_sc, bins=edges_corr)
        bar_offset_c = 0.25 / n_bins_c; bar_width_c = 2 * bar_offset_c
        ax = axes[0, 1]
        ax.bar(x_corr + bar_offset_c, n1_sc, width=bar_width_c, color='g', edgecolor='none')
        ax.bar(x_corr - bar_offset_c, n2_sc, width=bar_width_c, color='r', edgecolor='none')
        ax.set_xlim(0, 1); ax.legend(loc='upper left', frameon=False)
        ax.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
    else:
        axes[0, 1].axis('off')

    # subplot(2,2,3): Centroid model
    ax_m = axes[1, 0]
    p_same = float(centroid_distances_model_parameters[0])
    bw_full = (x_dist[1] - x_dist[0]) if number_of_bins > 1 else 1.0
    ax_m.bar(x_dist, centroid_distances_distribution, width=bw_full, color='b', edgecolor='none', label='Observed data')
    # Plot same/diff first, then overall, then same/diff AGAIN on top (MATLAB does this)
    ax_m.plot(x_dist, p_same * centroid_distances_model_same_cells, '--', color='g', linewidth=3, label='Same cell model')
    ax_m.plot(x_dist, (1 - p_same) * centroid_distances_model_different_cells, '--', color='r', linewidth=3, label='Different cells model')
    ax_m.plot(x_dist, centroid_distances_model_weighted_sum, '-', color='k', linewidth=3, label='Overall model')
    # Re-plot on top so dashed lines are visible over the solid black
    ax_m.plot(x_dist, p_same * centroid_distances_model_same_cells, '--', color='g', linewidth=3)
    ax_m.plot(x_dist, (1 - p_same) * centroid_distances_model_different_cells, '--', color='r', linewidth=3)
    if centroid_distance_intersection is not None and np.isfinite(centroid_distance_intersection):
        ci = float(centroid_distance_intersection)
        ymax_d = float(np.max(centroid_distances_distribution)) if len(centroid_distances_distribution) else 1
        ax_m.plot([ci, ci], [0, ymax_d], '--', color='k', linewidth=2)
        # Percentage annotations (MATLAB lines 915-922)
        norm_same = centroid_distances_model_same_cells / max(centroid_distances_model_same_cells.sum(), 1e-12)
        norm_diff = centroid_distances_model_different_cells / max(centroid_distances_model_different_cells.sum(), 1e-12)
        same_above = float(norm_same[x_dist > ci].sum())
        diff_above = float(norm_diff[x_dist > ci].sum())
        ax_m.text(ci + 1, 0.9 * ymax_d, f'{round(100 * same_above)}%',
                  fontsize=14, fontweight='bold', ha='center', color='g')
        ax_m.text(ci - 1, 0.9 * ymax_d, f'{round(100 * (1 - same_above))}%',
                  fontsize=14, fontweight='bold', ha='center', color='g')
        ax_m.text(ci + 1, 0.8 * ymax_d, f'{round(100 * diff_above)}%',
                  fontsize=14, fontweight='bold', ha='center', color='r')
        ax_m.text(ci - 1, 0.8 * ymax_d, f'{round(100 * (1 - diff_above))}%',
                  fontsize=14, fontweight='bold', ha='center', color='r')
    ax_m.set_xlim(0, microns_per_pixel * maximal_distance)
    ax_m.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
    ax_m.set_ylabel('Probability density', fontweight='bold', fontsize=14)

    # subplot(2,2,4): Spatial model
    if has_spatial:
        ax_ms = axes[1, 1]
        p_same_s = float(spatial_correlations_model_parameters[0])
        bw_full_c = (x_corr[1] - x_corr[0]) if n_bins_c > 1 else 0.05
        ax_ms.bar(x_corr, spatial_correlations_distribution, width=bw_full_c, color='b', edgecolor='none', label='Observed data')
        ax_ms.plot(x_corr, p_same_s * spatial_correlations_model_same_cells, '--', color='g', linewidth=3, label='Same cell model')
        ax_ms.plot(x_corr, (1-p_same_s) * spatial_correlations_model_different_cells, '--', color='r', linewidth=3, label='Different cells model')
        ax_ms.plot(x_corr, spatial_correlations_model_weighted_sum, '-', color='k', linewidth=3, label='Overall model')
        # Re-plot on top
        ax_ms.plot(x_corr, p_same_s * spatial_correlations_model_same_cells, '--', color='g', linewidth=3)
        ax_ms.plot(x_corr, (1-p_same_s) * spatial_correlations_model_different_cells, '--', color='r', linewidth=3)
        if spatial_correlation_intersection is not None and np.isfinite(spatial_correlation_intersection):
            sci = float(spatial_correlation_intersection)
            ymax_s = float(np.max(spatial_correlations_distribution)) if len(spatial_correlations_distribution) else 1
            ax_ms.plot([sci, sci], [0, ymax_s], '--', color='k', linewidth=2)
            # Percentage annotations (MATLAB lines 884-891)
            norm_same_s = spatial_correlations_model_same_cells / max(spatial_correlations_model_same_cells.sum(), 1e-12)
            norm_diff_s = spatial_correlations_model_different_cells / max(spatial_correlations_model_different_cells.sum(), 1e-12)
            same_above_s = float(norm_same_s[x_corr > sci].sum())
            diff_above_s = float(norm_diff_s[x_corr > sci].sum())
            ax_ms.text(sci + 0.1, 0.9 * ymax_s, f'{round(100 * same_above_s)}%',
                       fontsize=14, fontweight='bold', ha='center', color='g')
            ax_ms.text(sci - 0.1, 0.9 * ymax_s, f'{round(100 * (1 - same_above_s))}%',
                       fontsize=14, fontweight='bold', ha='center', color='g')
            ax_ms.text(sci + 0.1, 0.8 * ymax_s, f'{round(100 * diff_above_s)}%',
                       fontsize=14, fontweight='bold', ha='center', color='r')
            ax_ms.text(sci - 0.1, 0.8 * ymax_s, f'{round(100 * (1 - diff_above_s))}%',
                       fontsize=14, fontweight='bold', ha='center', color='r')
        ax_ms.set_xlim(0, 1)
        ax_ms.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
        ax_ms.set_ylabel('Probability density', fontweight='bold', fontsize=14)
        ax_ms.legend(loc='upper left', frameon=False)
    else:
        axes[1, 1].axis('off')

    savefig_both(fig, os.path.join(out_dir, "Stage 3 - model"), also_pdf=also_pdf, show=show)
    if not INLINE_PLOTS and not show:
        plt.close(fig)


def plot_cell_scores(
    cell_scores, cell_scores_exclusive, cell_scores_positive, cell_scores_negative,
    p_same_registered_pairs, out_dir, show=False, also_pdf=False,
):
    """Plots score distributions (Stage 5)."""
    xout_temp = np.linspace(0, 1, 41)
    xout = xout_temp[1::2]  # 20 bin centers
    step = xout[1] - xout[0] if len(xout) > 1 else 0.05
    edges = np.concatenate([xout - step/2, [xout[-1] + step/2]])
    number_of_clusters = len(cell_scores) if cell_scores is not None else 0
    size_x, size_y = 0.65, 0.65

    def _score_panel(fig, pos, inset_pos, data, xlabel_text, show_title=False):
        ax = fig.add_axes(pos)
        if data is None or len(data) == 0: return
        n1, _ = np.histogram(data, bins=edges)
        total = float(np.sum(n1))
        if total > 0: n1 = n1 / total
        ax.bar(xout, n1, width=step, color='steelblue')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        xtk = np.linspace(0, 1, 6)
        ax.set_xticks(xtk); ax.set_xticklabels([f'{v:.1f}' for v in xtk], fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel_text, fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
        if show_title:
            ax.text(-0.25, 1.2, f'{number_of_clusters} registered cells', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
        ax_in = fig.add_axes(inset_pos)
        ax_in.plot(np.flip(xout), np.cumsum(np.flip(n1)), linewidth=2)
        ax_in.set_ylim(0, 1); ax_in.invert_xaxis()
        xtk3 = np.linspace(0, 1, 3)
        ax_in.set_xticks(xtk3); ax_in.set_xticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
        ax_in.set_yticks(xtk3); ax_in.set_yticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
        ax_in.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax_in.set_ylabel('Cum. fraction', fontsize=14, fontweight='bold')

    fig = plt.figure(figsize=(12, 10))
    sx2, sy2 = size_x/2, size_y/2
    _score_panel(fig, [0.12, 0.58, sx2, sy2], [0.2, 0.73, sx2/3, sy2/3], cell_scores_negative, 'True negative scores')
    _score_panel(fig, [0.6, 0.58, sx2, sy2], [0.68, 0.73, sx2/3, sy2/3], cell_scores_positive, 'True positive scores', show_title=True)
    _score_panel(fig, [0.12, 0.1, sx2, sy2], [0.2, 0.25, sx2/3, sy2/3], cell_scores_exclusive, 'Exclusivity cell scores')
    _score_panel(fig, [0.6, 0.1, sx2, sy2], [0.68, 0.25, sx2/3, sy2/3], cell_scores, 'Overall cell scores')
    savefig_both(fig, os.path.join(out_dir, "Stage 5 - cell scores"), also_pdf=also_pdf, show=show)
    if not INLINE_PLOTS and not show: plt.close(fig)

    # P_same pairs plot
    if p_same_registered_pairs is not None:
        p_pairs = []
        if isinstance(p_same_registered_pairs, (list, tuple)):
            for mat in p_same_registered_pairs:
                if mat is not None:
                    mat_arr = np.asarray(mat)
                    if mat_arr.ndim == 2:
                        rows_m, cols_m = mat_arr.shape
                        for k in range(rows_m):
                            for m in range(k+1, cols_m):
                                v = mat_arr[k, m]
                                if np.isfinite(v): p_pairs.append(v)
                    else:
                        v = mat_arr.ravel(); p_pairs.extend(v[np.isfinite(v)])
        elif isinstance(p_same_registered_pairs, np.ndarray):
            v = p_same_registered_pairs.ravel(); p_pairs = list(v[np.isfinite(v)])
        if len(p_pairs) > 0:
            p_pairs = np.array(p_pairs)
            fig2 = plt.figure(figsize=(6, 5))
            ax2 = fig2.add_axes([0.15, 0.15, 0.75, 0.75])
            n1_p, _ = np.histogram(p_pairs, bins=edges)
            total_p = float(np.sum(n1_p))
            if total_p > 0: n1_p = n1_p / total_p
            ax2.bar(xout, n1_p, width=step, color='steelblue')
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            xtk = np.linspace(0, 1, 6)
            ax2.set_xticks(xtk); ax2.set_xticklabels([f'{v:.1f}' for v in xtk], fontsize=14, fontweight='bold')
            ax2.set_xlabel('Registered pairs P$_{same}$', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Probability', fontsize=14, fontweight='bold')
            ax2_in = fig2.add_axes([0.3, 0.5, 0.3, 0.3])
            ax2_in.plot(np.flip(xout), np.cumsum(np.flip(n1_p)), linewidth=2)
            ax2_in.set_ylim(0, 1); ax2_in.invert_xaxis()
            xtk3 = np.linspace(0, 1, 3)
            ax2_in.set_xticks(xtk3); ax2_in.set_xticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
            ax2_in.set_yticks(xtk3); ax2_in.set_yticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
            ax2_in.set_xlabel('P$_{same}$', fontsize=14, fontweight='bold')
            ax2_in.set_ylabel('Cum. fraction', fontsize=14, fontweight='bold')
            savefig_both(fig2, os.path.join(out_dir, "Stage 5 - Registered pairs P_same"), also_pdf=also_pdf, show=show)
            if not INLINE_PLOTS and not show: plt.close(fig2)


def plot_all_registered_projections(
    spatial_footprints, cell_to_index_map, out_dir, show=False, also_pdf=False, stage_label="Stage 5",
):
    """Plots projections of all cells. Green = cells in all sessions."""
    if spatial_footprints is None or cell_to_index_map is None:
        return
    if isinstance(spatial_footprints, np.ndarray) and spatial_footprints.dtype == object:
        fp_list = [np.asarray(e) for e in spatial_footprints.tolist()]
    elif isinstance(spatial_footprints, (list, tuple)):
        fp_list = [np.asarray(e) for e in spatial_footprints]
    else:
        return

    map_arr = np.asarray(cell_to_index_map)
    num_sessions = len(fp_list)
    if map_arr.shape[1] != num_sessions: return

    present_counts = np.sum(map_arr > 0, axis=1)
    idx_all = np.where(present_counts == num_sessions)[0]
    pixel_weight_threshold = 0.5

    projections = []
    for s in range(num_sessions):
        fps = fp_list[s]
        if fps.ndim != 3: continue
        # Convert 1-based to 0-based
        c_idxs = map_arr[:, s].astype(int).copy()
        absent = c_idxs <= 0
        c_idxs -= 1
        c_idxs[absent] = -1

        h, w = fps.shape[1], fps.shape[2]
        normalized_fps = np.zeros_like(fps, dtype=float)
        for k in range(fps.shape[0]):
            fp = fps[k].astype(float)
            mx_fp = float(np.max(fp))
            if mx_fp > 0:
                fp[fp < pixel_weight_threshold * mx_fp] = 0
                normalized_fps[k] = fp / mx_fp

        idxs_all = c_idxs[idx_all]; idxs_all = idxs_all[idxs_all >= 0]
        other_mask = (c_idxs >= 0) & (present_counts < num_sessions)
        idxs_other = c_idxs[other_mask]; idxs_other = idxs_other[idxs_other >= 0]

        img_rgb = np.zeros((h, w, 3), dtype=float)
        sum_all = np.sum(normalized_fps[idxs_all], axis=0) if len(idxs_all) > 0 else np.zeros((h, w))
        sum_other = np.sum(normalized_fps[idxs_other], axis=0) if len(idxs_other) > 0 else np.zeros((h, w))
        img_rgb[..., 0] = sum_other
        img_rgb[..., 1] = sum_other + sum_all
        img_rgb[..., 2] = sum_other
        img_rgb = _clamp01(img_rgb)
        projections.append(img_rgb)

    if num_sessions <= 4:
        fig, axes = plt.subplots(1, num_sessions, figsize=(4 * num_sessions, 5))
        if num_sessions == 1: axes = [axes]
    else:
        subx = 4; suby = int(np.ceil(num_sessions / subx))
        fig, axes = plt.subplots(suby, subx, figsize=(4 * subx, 4 * suby))
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(len(axes_flat)):
        if i < num_sessions:
            axes_flat[i].imshow(projections[i])
            axes_flat[i].set_xticks([]); axes_flat[i].set_yticks([])
            axes_flat[i].set_title(f'Session {i + 1}', fontsize=14, fontweight='bold')
            if i == 0:
                h_img, w_img = projections[i].shape[0], projections[i].shape[1]
                axes_flat[i].text(0.01*w_img, 0.02*h_img, 'Detected in', fontsize=14, color='g', fontweight='bold')
                axes_flat[i].text(0.01*w_img, 0.06*h_img, 'all sessions', fontsize=14, color='g', fontweight='bold')
        else:
            axes_flat[i].axis('off')
    fname = f"{stage_label} - projections - {'initial' if '4' in stage_label else 'final'} registration"
    savefig_both(fig, os.path.join(out_dir, fname), also_pdf=also_pdf, show=show)
    if not INLINE_PLOTS and not show: plt.close(fig)


def plot_init_registration(
    cell_to_index_map, number_of_bins, spatial_footprints, initial_registration_type,
    registered_cells, non_registered_cells, microns_per_pixel=1.0, maximal_distance=10.0,
    out_dir=".", show=False, also_pdf=False,
):
    """Plots initial registration results (Stage 4)."""
    fig = plt.figure(figsize=(8, 6))
    if initial_registration_type.lower().startswith('spatial'):
        xout = np.linspace(0, 1, number_of_bins)
        step = xout[1] - xout[0] if len(xout) > 1 else 0.05
        edges = np.concatenate([xout - step/2, [xout[-1] + step/2]])
        n1, _ = np.histogram(registered_cells, bins=edges)
        n2, _ = np.histogram(non_registered_cells, bins=edges)
        bar_offset = 0.25 / number_of_bins; bar_width = 2 * bar_offset
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        ax.bar(xout + bar_offset, n1, width=bar_width, color='g', edgecolor='none', label='Same Cell')
        ax.bar(xout - bar_offset, n2, width=bar_width, color='r', edgecolor='none', label='Different Cells')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
    else:
        xout = np.linspace(0, maximal_distance, number_of_bins)
        step = xout[1] - xout[0] if len(xout) > 1 else 1.0
        edges = np.concatenate([xout - step/2, [xout[-1] + step/2]])
        n1, _ = np.histogram(registered_cells, bins=edges)
        n2, _ = np.histogram(non_registered_cells, bins=edges)
        bar_offset = 0.25 * microns_per_pixel * maximal_distance / number_of_bins; bar_width = 2 * bar_offset
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        ax.bar(microns_per_pixel*xout + bar_offset, n1, width=bar_width, color='g', edgecolor='none', label='Same Cell')
        ax.bar(microns_per_pixel*xout - bar_offset, n2, width=bar_width, color='r', edgecolor='none', label='Different Cells')
        ax.set_xlim(0, microns_per_pixel * maximal_distance)
        ax.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14); ax.legend(loc='upper left', frameon=False)
    savefig_both(fig, os.path.join(out_dir, "Stage 4 - same versus different cells"), also_pdf=also_pdf, show=show)
    if not INLINE_PLOTS and not show: plt.close(fig)
    plot_all_registered_projections(spatial_footprints, cell_to_index_map, out_dir, show=show, also_pdf=also_pdf, stage_label="Stage 4")


# ============================================================================ #
#                           CONFIGURATION                                      #
# ============================================================================ #
# ---- Set your folder path here (same as MATLAB validate_alignment.m) ----
folder_path = r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_superalignment\L612_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN"
#folder_path = r"Z:\John\Subjects - GCaMP Recordings\L613_F_LeftPFC_L6Chrimson_PFCgcamp8f_Panrec"

# ---- Parameters (matching batchRunCellReg_ULTIMATE.m lines 102-131) ----
microns_per_pixel = 2
maximal_distance = 15                 # micrometers
p_same_certainty_threshold = 0.95
p_same_threshold = 0.5
registration_approach = 'Probabilistic'
alignment_type = 'Translations and Rotations'
sufficient_correlation_centroids = 0.2
sufficient_correlation_footprints = 0.3
correlation_threshold = 0.65
alignable_threshold = 0.3             # minimum mean-image correlation to attempt alignment

# ---- Display / save options ----
SHOW_FIGURES = True                   # True = display figures in real time
SAVE_FIGURES = True                   # True = save PNGs to disk
# NOTE: INLINE_PLOTS is set near top of file (line ~59)
# ============================================================================ #


def _extract_p_from_model_string(model_string: str) -> float:
    """Extract the mixing weight p from the best_model_string."""
    try:
        return float(model_string.split(',')[0].split('=')[1])
    except Exception:
        return 0.5


def _compute_histogram_distribution(data: np.ndarray, bin_centers: np.ndarray,
                                     number_of_bins: int,
                                     scale: float = 1.0) -> np.ndarray:
    """Compute MATLAB-style normalized histogram distribution for model plotting."""
    d = np.asarray(data, dtype=np.float64)
    d = d[np.isfinite(d)]
    centers = np.asarray(bin_centers, dtype=np.float64)
    if len(centers) >= 2:
        step_c = centers[1] - centers[0]
        edges = np.empty(len(centers) + 1)
        edges[0] = centers[0] - step_c / 2
        for i in range(1, len(centers)):
            edges[i] = (centers[i - 1] + centers[i]) / 2
        edges[-1] = centers[-1] + step_c / 2
        counts, _ = np.histogram(d, bins=edges)
    else:
        counts = np.array([len(d)])
    counts = counts.astype(np.float64)
    step = scale * (centers[1] - centers[0]) if len(centers) > 1 else 1.0
    rng = scale * (centers[-1] - centers[0]) if len(centers) > 1 else 1.0
    denom = step + rng
    total = max(counts.sum(), 1.0)
    dist = counts / total * (number_of_bins / denom)
    return dist


def plot_pairwise_session_overlap(
    spatial_footprints,
    cell_to_index_map,
    out_dir,
    show=False,
):
    """
    Pairwise session overlap matrix.
    N×N grid: Red = session i only, Green = session j only, Yellow = shared.
    Diagonal = all cells for that session in white.
    """
    if spatial_footprints is None or cell_to_index_map is None:
        return

    if isinstance(spatial_footprints, np.ndarray) and spatial_footprints.dtype == object:
        fp_list = [np.asarray(e) for e in spatial_footprints.tolist()]
    elif isinstance(spatial_footprints, (list, tuple)):
        fp_list = [np.asarray(e) for e in spatial_footprints]
    else:
        return

    map_arr = np.asarray(cell_to_index_map)  # (n_clusters, n_sessions), 1-indexed, 0=absent
    n_sessions = len(fp_list)
    if map_arr.shape[1] != n_sessions:
        print(f"Warning: map sessions ({map_arr.shape[1]}) != footprint sessions ({n_sessions})")
        return

    pixel_weight_threshold = 0.5

    # Pre-normalize footprints
    norm_fps = []
    for s in range(n_sessions):
        fps = fp_list[s].astype(float)
        out = np.zeros_like(fps)
        for k in range(fps.shape[0]):
            mx = float(np.max(fps[k]))
            if mx > 0:
                fp = fps[k].copy()
                fp[fp < pixel_weight_threshold * mx] = 0
                out[k] = fp / mx
        norm_fps.append(out)

    # Convert 1-based map to 0-based: 0 (absent) → -1, positive → subtract 1
    def _to_0based(s):
        c = map_arr[:, s].astype(int).copy()
        absent = c <= 0
        c -= 1
        c[absent] = -1
        return c

    sess_idxs = [_to_0based(s) for s in range(n_sessions)]
    h, w = norm_fps[0].shape[1], norm_fps[0].shape[2]

    fig, axes = plt.subplots(n_sessions, n_sessions,
                             figsize=(4 * n_sessions, 4 * n_sessions))
    fig.suptitle('Pairwise Session Cell Overlap', fontsize=18, fontweight='bold', y=0.98)

    for i in range(n_sessions):
        for j in range(n_sessions):
            ax = axes[i][j] if n_sessions > 1 else axes

            if i == j:
                # Diagonal: all cells in white
                cells = sess_idxs[i][sess_idxs[i] >= 0]
                proj = np.zeros((h, w), dtype=float)
                if len(cells) > 0:
                    proj = np.clip(np.sum(norm_fps[i][cells], axis=0), 0, 1)
                ax.imshow(np.stack([proj, proj, proj], axis=-1))
                ax.set_title(f'Session {i+1}\n({len(cells)} cells)',
                             fontsize=12, fontweight='bold')
            else:
                # Off-diagonal: i (red) vs j (green), shared = yellow
                both = (sess_idxs[i] >= 0) & (sess_idxs[j] >= 0)
                only_i = (sess_idxs[i] >= 0) & (sess_idxs[j] < 0)
                only_j = (sess_idxs[i] < 0) & (sess_idxs[j] >= 0)

                # Shared cells projection (average both sessions' footprints)
                shared_proj = np.zeros((h, w), dtype=float)
                if both.sum() > 0:
                    si = sess_idxs[i][both]
                    sj = sess_idxs[j][both]
                    shared_proj = np.clip(
                        np.sum(norm_fps[i][si], axis=0) +
                        np.sum(norm_fps[j][sj], axis=0), 0, 1)

                # Unique projections
                proj_i = np.zeros((h, w), dtype=float)
                if only_i.sum() > 0:
                    proj_i = np.clip(np.sum(norm_fps[i][sess_idxs[i][only_i]], axis=0), 0, 1)
                proj_j = np.zeros((h, w), dtype=float)
                if only_j.sum() > 0:
                    proj_j = np.clip(np.sum(norm_fps[j][sess_idxs[j][only_j]], axis=0), 0, 1)

                # RGB: shared=yellow, i-only=red, j-only=green
                r_ch = np.clip(proj_i + shared_proj, 0, 1)
                g_ch = np.clip(proj_j + shared_proj, 0, 1)
                ax.imshow(np.stack([r_ch, g_ch, np.zeros((h, w))], axis=-1))

                n_shared = int(both.sum())
                n_oi = int(only_i.sum())
                n_oj = int(only_j.sum())
                ax.set_title(f'S{i+1} vs S{j+1}\n{n_shared} shared',
                             fontsize=11, fontweight='bold')
                ax.text(3, h - 8, f'S{i+1} only: {n_oi}',
                        fontsize=9, color='red', fontweight='bold', va='bottom')
                ax.text(3, h - 22, f'S{j+1} only: {n_oj}',
                        fontsize=9, color='lime', fontweight='bold', va='bottom')

            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'Session {i+1}', fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_both(fig, os.path.join(out_dir, "Stage 5 - pairwise session overlap"), show=show)
    if not INLINE_PLOTS and not show:
        plt.close(fig)


def run_pipeline(folder_path: str):
    """Run the full CellReg validation pipeline matching MATLAB's
    batchRunCellReg(folder_path, [], [], true)."""

    # Version check — confirm this is the latest synced file
    print(f"[SYNC CHECK] INLINE_PLOTS={INLINE_PLOTS}, matplotlib backend={matplotlib.get_backend()}")

    normalized_maximal_distance = maximal_distance / microns_per_pixel

    print("=" * 60)
    print("  Python CellReg Validation Pipeline")
    print("  (Equivalent to MATLAB batchRunCellReg validation_mode)")
    print("=" * 60)

    # ================================================================== #
    #  Find session folders (MATLAB lines 151-157)
    # ================================================================== #
    plane0_folders = list_session_folders(Path(folder_path))
    print(f"\nFound {len(plane0_folders)} session folders")

    # Validation mode: first 4 sessions only (MATLAB line 157)
    plane0_folders = plane0_folders[:4]
    n_sessions = len(plane0_folders)
    print(f"Validation mode: using first {n_sessions} sessions")

    # Find CellReg.mat files (MATLAB lines 162-176)
    sess_fovs = get_cellreg_files(plane0_folders)
    print(f"Found {len(sess_fovs)} CellReg.mat files")
    if len(sess_fovs) < 2:
        raise RuntimeError("Need at least 2 sessions with CellReg.mat files.")

    # ================================================================== #
    #  Setup directories (MATLAB lines 186-192)
    # ================================================================== #
    results_root = Path(folder_path) / "1_CellReg"
    fov_dir = results_root / "FOV1"
    figures_directory = fov_dir / "Figures"
    results_directory = fov_dir / "Results"
    if SAVE_FIGURES:
        for d in [results_root, fov_dir, figures_directory, results_directory]:
            d.mkdir(parents=True, exist_ok=True)
    fig_dir = str(figures_directory) if SAVE_FIGURES else None

    # ================================================================== #
    #  STAGE 1 — Load spatial footprints + projections (MATLAB lines 206-219)
    # ================================================================== #
    test_run = True
    if test_run:
        import random
        sess_fovs = random.sample(sess_fovs,4)

    print("\n--- Stage 1: Loading spatial footprints ---")
    spatial_footprints_raw = []
    for sf in sess_fovs:
        fp = get_spatial_footprints(sf)
        spatial_footprints_raw.append(fp)
        print(f"  {Path(sf).parent.parent.parent.name}: "
              f"{fp.shape[0]} cells, {fp.shape[1]}×{fp.shape[2]} FOV")

    # Compute projections (before normalize/adjust)
    footprints_projections = compute_footprint_projections(spatial_footprints_raw)

    # FIGURE: Stage 1 — spatial footprints projections
    plot_session_projections(footprints_projections, fig_dir, show=SHOW_FIGURES)
    print("  ✓ Stage 1 figure done")

    # ================================================================== #
    #  Load mean images with drift correction (MATLAB lines 226-254)
    # ================================================================== #
    print("\n--- Loading mean images ---")
    mean_images = []
    for pf in sess_fovs:
        mi = get_mean_image(Path(pf).parent, apply_drift_correction=True)
        mean_images.append(mi)
        print(f"  {Path(pf).parent.parent.name}: {mi.shape}")

    # ================================================================== #
    #  Normalize + Adjust FOV (MATLAB lines 392-407)
    # ================================================================== #
    print("\n--- Normalizing and adjusting FOV ---")
    normalized_fps = normalize_footprints(spatial_footprints_raw)
    adjusted_fps, adjusted_fov, adj_x, adj_y, padding = adjust_fov_size(normalized_fps)
    del normalized_fps

    # Compute adjusted projections and centroids
    adjusted_projections = compute_footprint_projections(adjusted_fps)
    centroid_locations = compute_centroids(adjusted_fps, microns_per_pixel)
    centroid_projections = compute_centroid_projections(centroid_locations, adjusted_fps)

    # ================================================================== #
    #  STAGE 2 — Align using mean images (MATLAB lines 418-439)
    # ================================================================== #
    print("\n--- Stage 2: Aligning to reference session ---")
    reference_session_index = 0  # MATLAB default for validation

    aligner = MeanImageAligner()
    aligned_fps = [None] * n_sessions
    aligned_centroid_locations = [None] * n_sessions
    alignment_translations = np.zeros((3, n_sessions))
    maximal_cross_correlation = np.zeros(n_sessions)

    for i in range(n_sessions):
        if i == reference_session_index:
            aligned_fps[i] = adjusted_fps[i]
            aligned_centroid_locations[i] = centroid_locations[i]
            maximal_cross_correlation[i] = 1.0
        else:
            _, method, peak, tform, _, _ = aligner.align(
                mean_images[reference_session_index],
                mean_images[i],
                filter_mode='highpass',
                outlier_mode='off'
            )
            maximal_cross_correlation[i] = peak

            if tform is not None and peak >= alignable_threshold:
                from skimage import transform as sktransform

                # Store alignment parameters
                params = tform.params
                alignment_translations[0, i] = params[0, 2]       # x-translation
                alignment_translations[1, i] = params[1, 2]       # y-translation
                angle_rad = np.arctan2(params[1, 0], params[0, 0])
                alignment_translations[2, i] = np.degrees(angle_rad)  # rotation

                # Warp all spatial footprints
                n_cells = adjusted_fps[i].shape[0]
                aligned = np.zeros_like(adjusted_fps[i])
                for c in range(n_cells):
                    aligned[c] = sktransform.warp(
                        adjusted_fps[i][c],
                        tform.inverse,
                        output_shape=adjusted_fps[i][c].shape,
                        order=1,
                        preserve_range=True,
                        mode='constant',
                        cval=0.0,
                    )
                aligned_fps[i] = aligned

                # Warp centroids
                cents = centroid_locations[i]
                if len(cents) > 0:
                    coords = np.column_stack([cents[:, 0], cents[:, 1], np.ones(len(cents))])
                    transformed = (tform.params @ coords.T).T
                    aligned_centroid_locations[i] = transformed[:, :2]
                else:
                    aligned_centroid_locations[i] = cents

                print(f"  Session {i+1}: peak={peak:.3f}, "
                      f"dx={alignment_translations[0,i]:.1f}, "
                      f"dy={alignment_translations[1,i]:.1f}, "
                      f"rot={alignment_translations[2,i]:.2f}°")
            else:
                aligned_fps[i] = adjusted_fps[i]
                aligned_centroid_locations[i] = centroid_locations[i]
                print(f"  Session {i+1}: peak={peak:.3f} (below threshold)")

    # Compute corrected projections (after alignment)
    corrected_projections = compute_footprint_projections(aligned_fps)

    # FIGURE: Alignment validation deck (overview + per-session panels)
    session_names = [str(pf) for pf in plane0_folders]
    validate_alignment_deck(
        mean_images,
        adjusted_projections,      # pre-alignment
        corrected_projections,     # post-alignment
        reference_session_index=reference_session_index,
        alignment_translations=alignment_translations,
        scores=maximal_cross_correlation,
        out_dir=fig_dir,
        session_names=session_names,
        show=SHOW_FIGURES,
    )
    print("  ✓ Alignment validation figures done")

    # ================================================================== #
    #  STAGE 2b — HP / LP / BP Filter Visualization
    # ================================================================== #
    print("\n--- Stage 2b: Filter visualization for alignment diagnostics ---")
    from scipy.ndimage import gaussian_filter as _gf

    blur_hp = 12.0   # high-pass sigma (matches CellRegConfig default)
    blur_lp = 5.0    # low-pass sigma
    blur_bp1 = 2.0   # band-pass sigma (small)
    blur_bp2 = 12.0  # band-pass sigma (large)

    # --- Per-session HP/LP/BP filter views ---
    fig_filt, axes_filt = plt.subplots(n_sessions, 4, figsize=(20, 5 * n_sessions))
    if n_sessions == 1:
        axes_filt = axes_filt[np.newaxis, :]

    for i in range(n_sessions):
        img = mean_images[i].astype(np.float32)
        img_safe = np.nan_to_num(img, nan=0.0)
        hp = img_safe - _gf(img_safe, blur_hp)
        lp = _gf(img_safe, blur_lp)
        bp = _gf(img_safe, blur_bp1) - _gf(img_safe, blur_bp2)

        for ax_idx, (data, title) in enumerate([
            (img_safe, f'Session {i+1}: Raw'),
            (hp,       f'Session {i+1}: High-Pass (σ={blur_hp})'),
            (lp,       f'Session {i+1}: Low-Pass (σ={blur_lp})'),
            (bp,       f'Session {i+1}: Band-Pass ({blur_bp1}-{blur_bp2})'),
        ]):
            ax = axes_filt[i, ax_idx]
            vmin, vmax = np.nanpercentile(data, [1, 99])
            if vmax <= vmin:
                vmax = vmin + 1
            ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

    fig_filt.suptitle('Stage 2b — Filtered Mean Images', fontsize=16, fontweight='bold')
    fig_filt.tight_layout(rect=[0, 0, 1, 0.96])
    if fig_dir:
        savefig_both(fig_filt, os.path.join(fig_dir, "Stage 2b - filtered mean images"),
                     show=SHOW_FIGURES)
    if not INLINE_PLOTS and not SHOW_FIGURES:
        plt.close(fig_filt)

    # --- Pairwise HP-filtered overlay: ref vs each session (pre and post alignment) ---
    ref_img = mean_images[reference_session_index].astype(np.float32)
    ref_safe = np.nan_to_num(ref_img, nan=0.0)
    ref_hp = ref_safe - _gf(ref_safe, blur_hp)

    other_indices = [i for i in range(n_sessions) if i != reference_session_index]
    if other_indices:
        fig_ov, axes_ov = plt.subplots(len(other_indices), 4,
                                       figsize=(20, 5 * len(other_indices)))
        if len(other_indices) == 1:
            axes_ov = axes_ov[np.newaxis, :]

        for row, si in enumerate(other_indices):
            # Pre-alignment HP
            mov_raw = mean_images[si].astype(np.float32)
            mov_safe = np.nan_to_num(mov_raw, nan=0.0)
            mov_hp = mov_safe - _gf(mov_safe, blur_hp)

            # Post-alignment HP (apply the same transform that was used)
            from skimage import transform as _sktf
            if maximal_cross_correlation[si] >= alignable_threshold:
                # Reconstruct the aligned mean image using the stored tform
                _, _, _, tform_i, _, _ = aligner.align(
                    mean_images[reference_session_index],
                    mean_images[si],
                    filter_mode='highpass',
                    outlier_mode='off'
                )
                if tform_i is not None:
                    aligned_mi = _sktf.warp(
                        mov_safe, tform_i.inverse, output_shape=ref_safe.shape,
                        order=1, preserve_range=True, mode='constant', cval=0.0
                    )
                else:
                    aligned_mi = mov_safe
            else:
                aligned_mi = mov_safe
            aligned_hp = aligned_mi - _gf(aligned_mi, blur_hp)

            # Normalize for overlay
            def _norm_hp(x):
                lo, hi = np.percentile(x, [2, 98])
                if hi <= lo: return np.zeros_like(x)
                return np.clip((x - lo) / (hi - lo), 0, 1)

            rn = _norm_hp(ref_hp)
            mn = _norm_hp(mov_hp)
            an = _norm_hp(aligned_hp)

            # Pre-alignment overlay (cyan=ref, red=moving)
            pre_overlay = np.zeros((*ref_hp.shape, 3))
            pre_overlay[..., 0] = mn
            pre_overlay[..., 1] = rn
            pre_overlay[..., 2] = rn

            # Post-alignment overlay
            post_overlay = np.zeros((*ref_hp.shape, 3))
            post_overlay[..., 0] = an
            post_overlay[..., 1] = rn
            post_overlay[..., 2] = rn

            # Pre-alignment raw correlation
            both_valid_pre = np.isfinite(ref_img) & np.isfinite(mov_raw)
            corr_pre = np.corrcoef(ref_img[both_valid_pre].ravel(),
                                   mov_raw[both_valid_pre].ravel())[0, 1] if both_valid_pre.sum() > 100 else float('nan')

            # Post-alignment raw correlation
            both_valid_post = (aligned_mi > 0) & np.isfinite(ref_img)
            corr_post = np.corrcoef(ref_img[both_valid_post].ravel(),
                                    aligned_mi[both_valid_post].ravel())[0, 1] if both_valid_post.sum() > 100 else float('nan')

            titles = [
                f'S{si+1} HP (pre)',
                f'Overlay pre (r={corr_pre:.3f})',
                f'S{si+1} HP (post)',
                f'Overlay post (r={corr_post:.3f})',
            ]
            imgs = [mov_hp, pre_overlay, aligned_hp, post_overlay]
            for col, (data, title) in enumerate(zip(imgs, titles)):
                ax = axes_ov[row, col]
                if data.ndim == 2:
                    vmin, vmax = np.percentile(data, [2, 98])
                    if vmax <= vmin: vmax = vmin + 1
                    ax.imshow(data, cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    ax.imshow(np.clip(data, 0, 1))
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')

        fig_ov.suptitle('Stage 2b — HP-Filtered Alignment Overlays (Cyan=Ref, Red=Moving)',
                        fontsize=14, fontweight='bold')
        fig_ov.tight_layout(rect=[0, 0, 1, 0.96])
        if fig_dir:
            savefig_both(fig_ov, os.path.join(fig_dir,
                         "Stage 2b - HP alignment overlays"), show=SHOW_FIGURES)
        if not INLINE_PLOTS and not SHOW_FIGURES:
            plt.close(fig_ov)

    print("  ✓ Stage 2b filter visualization done")

    # ================================================================== #
    #  STAGE 3a — Data distribution (MATLAB lines 449-466)
    # ================================================================== #
    print("\n--- Stage 3a: Computing cell-pair similarity distributions ---")
    number_of_bins, _ = estimate_num_bins(adjusted_fps, normalized_maximal_distance)
    centers_of_bins = (
        np.linspace(0, normalized_maximal_distance, number_of_bins, dtype=np.float64),
        np.linspace(0, 1, number_of_bins, dtype=np.float64),
    )

    data_dist = compute_data_distribution(
        aligned_fps,
        aligned_centroid_locations,
        normalized_maximal_distance,
    )

    # FIGURE: Stage 3 — x,y displacements
    plot_x_y_displacements(
        data_dist['neighbors_x_displacements'],
        data_dist['neighbors_y_displacements'],
        microns_per_pixel,
        normalized_maximal_distance,
        number_of_bins,
        centers_of_bins,
        fig_dir,
        show=SHOW_FIGURES,
    )
    print("  ✓ Stage 3a displacement figure done")

    # ================================================================== #
    #  STAGE 3b — Probabilistic models (MATLAB lines 468-521)
    # ================================================================== #
    print("\n--- Stage 3b: Fitting probabilistic models ---")
    centroid_centers = centers_of_bins[0]
    corr_centers = centers_of_bins[1]

    # Centroid distances model
    (p_same_given_centroid_distance,
     centroid_same_model,
     centroid_diff_model,
     centroid_mixture_model,
     centroid_intersection,
     centroid_best_str,
     centroid_mse) = compute_centroid_distances_model_custom(
        data_dist['neighbors_centroid_distances'],
        number_of_bins,
        centers_of_bins,
        microns_per_pixel=microns_per_pixel,
    )
    p_centroid = _extract_p_from_model_string(centroid_best_str)
    print(f"  [DEBUG centroid] best_str = {centroid_best_str}")
    print(f"  [DEBUG centroid] p_centroid = {p_centroid}")
    print(f"  [DEBUG centroid] same_model  range: [{centroid_same_model.min():.6f}, {centroid_same_model.max():.6f}]")
    print(f"  [DEBUG centroid] diff_model  range: [{centroid_diff_model.min():.6f}, {centroid_diff_model.max():.6f}]")
    print(f"  [DEBUG centroid] mixture     range: [{centroid_mixture_model.min():.6f}, {centroid_mixture_model.max():.6f}]")
    print(f"  [DEBUG centroid] p*same      range: [{(p_centroid*centroid_same_model).min():.6f}, {(p_centroid*centroid_same_model).max():.6f}]")
    print(f"  [DEBUG centroid] (1-p)*diff  range: [{((1-p_centroid)*centroid_diff_model).min():.6f}, {((1-p_centroid)*centroid_diff_model).max():.6f}]")

    # Spatial correlations model
    (p_same_given_spatial_correlation,
     spatial_same_model,
     spatial_diff_model,
     spatial_mixture_model,
     spatial_intersection,
     spatial_best_str,
     spatial_mse) = compute_spatial_correlations_model(
        data_dist['neighbors_spatial_correlations'],
        number_of_bins,
        centers_of_bins,
    )
    p_spatial = _extract_p_from_model_string(spatial_best_str)

    # Compute histogram distributions for plotting
    centroid_distribution = _compute_histogram_distribution(
        data_dist['neighbors_centroid_distances'], centroid_centers, number_of_bins,
        scale=microns_per_pixel)
    spatial_distribution = _compute_histogram_distribution(
        data_dist['neighbors_spatial_correlations'][
            data_dist['neighbors_spatial_correlations'] >= 0  # filter negatives
        ],
        corr_centers, number_of_bins,
        scale=1.0,
    )

    # FIGURE: Stage 3 — models
    plot_models(
        # Centroid
        centroid_distances_model_parameters=np.array([p_centroid]),
        NN_centroid_distances=data_dist['NN_centroid_distances'],
        NNN_centroid_distances=data_dist['NNN_centroid_distances'],
        centroid_distances_distribution=centroid_distribution,
        centroid_distances_model_same_cells=centroid_same_model,
        centroid_distances_model_different_cells=centroid_diff_model,
        centroid_distances_model_weighted_sum=centroid_mixture_model,
        centroid_distance_intersection=centroid_intersection,
        centers_of_bins_dist=centroid_centers,
        # Spatial
        spatial_correlations_model_parameters=np.array([p_spatial]),
        NN_spatial_correlations=data_dist['NN_spatial_correlations'],
        NNN_spatial_correlations=data_dist['NNN_spatial_correlations'],
        spatial_correlations_distribution=spatial_distribution,
        spatial_correlations_model_same_cells=spatial_same_model,
        spatial_correlations_model_different_cells=spatial_diff_model,
        spatial_correlations_model_weighted_sum=spatial_mixture_model,
        spatial_correlation_intersection=spatial_intersection,
        centers_of_bins_corr=corr_centers,
        # General
        microns_per_pixel=microns_per_pixel,
        maximal_distance=normalized_maximal_distance,
        out_dir=fig_dir,
        show=SHOW_FIGURES,
    )
    print("  ✓ Stage 3b model figures done")

    # ================================================================== #
    #  Choose best model (MATLAB lines 504-508)
    # ================================================================== #
    best_model_string = choose_best_model(
        centroid_mse,
        spatial_mse,
        centroid_intersection=centroid_intersection,
        corr_intersection=spatial_intersection,
    )
    print(f"  Best model: {best_model_string}")

    # ================================================================== #
    #  Compute P_same (MATLAB lines 524-526)
    # ================================================================== #
    print("\n--- Computing P_same for all cell pairs ---")
    p_same_centroid, p_same_spatial = compute_p_same(
        data_dist['all_to_all_centroid_distances'],
        data_dist['all_to_all_spatial_correlations'],
        centers_of_bins,
        p_same_given_centroid_distance,
        p_same_given_spatial_correlation,
    )

    # ================================================================== #
    #  STAGE 4 — Initial registration (MATLAB lines 528-566)
    # ================================================================== #
    print("\n--- Stage 4: Initial registration ---")
    initial_registration_type = best_model_string

    if initial_registration_type == 'Spatial correlation':
        initial_threshold = spatial_intersection if np.isfinite(spatial_intersection) else 0.65
        (cell_to_index_map,
         registered_cells_metric,
         non_registered_cells_metric,
         _) = initial_registration_spatial_corr(
            aligned_fps,
            aligned_centroid_locations,
            maximal_distance=normalized_maximal_distance,
            spatial_correlation_threshold=initial_threshold,
        )
    else:  # Centroid distance
        initial_threshold = centroid_intersection if np.isfinite(centroid_intersection) else 5.0
        normalized_threshold = initial_threshold / microns_per_pixel
        (cell_to_index_map,
         registered_cells_metric,
         non_registered_cells_metric,
         _) = initial_registration_centroid_distances_custom(
            aligned_centroid_locations,
            maximal_distance=normalized_maximal_distance,
            centroid_distance_threshold=normalized_threshold,
        )

    print(f"  {cell_to_index_map.shape[0]} cell clusters found, "
          f"threshold={initial_threshold:.3f}")

    # FIGURE: Stage 4 — same vs different cells + initial projections
    plot_init_registration(
        cell_to_index_map,
        number_of_bins,
        aligned_fps,
        initial_registration_type,
        registered_cells_metric,
        non_registered_cells_metric,
        microns_per_pixel=microns_per_pixel,
        maximal_distance=normalized_maximal_distance,
        out_dir=fig_dir,
        show=SHOW_FIGURES,
    )
    print("  ✓ Stage 4 figures done")

    # ================================================================== #
    #  STAGE 5 — Final registration (MATLAB lines 573-645)
    # ================================================================== #
    print("\n--- Stage 5: Final registration (clustering) ---")

    if best_model_string == 'Spatial correlation':
        all_to_all_p_same = p_same_spatial
    else:
        all_to_all_p_same = p_same_centroid

    # cluster_cells_matlab returns: (map, centroids, cluster_scores_dict)
    (optimal_cell_to_index_map,
     registered_cells_centroids,
     cluster_scores_dict) = cluster_cells_matlab(
        cell_to_index_map=cell_to_index_map,
        all_to_all_p_same=all_to_all_p_same,
        all_to_all_indexes=data_dist['all_to_all_indexes'],
        maximal_distance=normalized_maximal_distance,
        registration_threshold=p_same_threshold,
        centroid_locations=aligned_centroid_locations,
        registration_approach=registration_approach,
        transform_data=False,
    )

    # estimate_registration_accuracy returns: (p_same_vec, p_diff_vec, scores)
    p_same_vec, p_diff_vec, accuracy_scores = estimate_registration_accuracy(
        optimal_cell_to_index_map,
        all_to_all_p_same,
        data_dist['all_to_all_indexes'],
        threshold=p_same_threshold,
    )

    # ---- Extract score breakdowns from cluster_cells_matlab output ----
    n_clusters_final = optimal_cell_to_index_map.shape[0]

    if cluster_scores_dict:
        cell_scores = cluster_scores_dict.get("cell_scores", np.zeros(n_clusters_final))
        cell_scores_positive = cluster_scores_dict.get("cell_scores_positive", np.zeros(n_clusters_final))
        cell_scores_negative = cluster_scores_dict.get("cell_scores_negative", np.zeros(n_clusters_final))
        cell_scores_exclusive = cluster_scores_dict.get("cell_scores_exclusive", np.zeros(n_clusters_final))
        p_same_registered_pairs = cluster_scores_dict.get("p_same_registered_pairs", [])
    else:
        cell_scores = np.zeros(n_clusters_final)
        cell_scores_positive = np.zeros(n_clusters_final)
        cell_scores_negative = np.zeros(n_clusters_final)
        cell_scores_exclusive = np.zeros(n_clusters_final)
        p_same_registered_pairs = []

    print(f"  {n_clusters_final} cell clusters in final registration")

    # FIGURE: Stage 5 — cell scores
    plot_cell_scores(
        cell_scores,
        cell_scores_exclusive,
        cell_scores_positive,
        cell_scores_negative,
        p_same_registered_pairs,
        fig_dir,
        show=SHOW_FIGURES,
    )
    print("  ✓ Stage 5 score figures done")

    # FIGURE: Stage 5 — final projections
    plot_all_registered_projections(
        aligned_fps,
        optimal_cell_to_index_map,
        fig_dir,
        show=SHOW_FIGURES,
        stage_label="Stage 5",
    )
    print("  ✓ Stage 5 projection figures done")

    # FIGURE: Stage 5 — pairwise session overlap
    plot_pairwise_session_overlap(
        aligned_fps,
        optimal_cell_to_index_map,
        fig_dir,
        show=SHOW_FIGURES,
    )
    print("  ✓ Pairwise session overlap figure done")

    # ================================================================== #
    #  STAGE 6 — Dual-model comparison (experimental)
    # ================================================================== #
    print("\n--- Stage 6: Dual-model comparison (experimental) ---")

    # --- Option 1: Consensus clustering ---
    (consensus_map,
     map_centroid_only,
     map_spatial_only,
     n_centroid,
     n_spatial) = cluster_cells_consensus(
        cell_to_index_map=cell_to_index_map,
        p_same_centroid=p_same_centroid,
        p_same_spatial=p_same_spatial,
        all_to_all_indexes=data_dist['all_to_all_indexes'],
        maximal_distance=normalized_maximal_distance,
        registration_threshold=p_same_threshold,
        centroid_locations=aligned_centroid_locations,
        registration_approach=registration_approach,
        verbose=True,
    )

    # --- Option 2: Combined p_same ---
    print("  [Combined] Computing p_centroid × p_spatial...")
    p_same_combined = combine_p_same(p_same_centroid, p_same_spatial)

    (combined_map,
     _combined_centroids,
     _combined_scores) = cluster_cells_matlab(
        cell_to_index_map=cell_to_index_map,
        all_to_all_p_same=p_same_combined,
        all_to_all_indexes=data_dist['all_to_all_indexes'],
        maximal_distance=normalized_maximal_distance,
        registration_threshold=p_same_threshold,
        centroid_locations=aligned_centroid_locations,
        registration_approach=registration_approach,
        transform_data=False,
        verbose=False,
    )
    print(f"  [Combined] {combined_map.shape[0]} clusters")

    # --- Helper for counting cells present in all sessions ---
    def _count_all_sessions(cmap, n_sess):
        if cmap.shape[0] == 0:
            return 0
        present = np.sum(cmap > 0, axis=1)
        return int(np.sum(present == n_sess))

    # --- Option 3: Centroid-primary with spatial floor filter ---
    # Logic: centroid distance drives matching; spatial correlation is only
    # used as a sanity-check veto.  This is the least conservative approach.
    spatial_corr_floor = 0.5   # ← tune this: reject matches below this correlation

    print(f"  [Centroid-primary] Clustering with centroid p_same only, "
          f"then vetoing spatial corr < {spatial_corr_floor}...")

    # Step 1: cluster using centroid-only p_same
    (centroid_primary_map,
     _cp_centroids,
     _cp_scores) = cluster_cells_matlab(
        cell_to_index_map=cell_to_index_map,
        all_to_all_p_same=p_same_centroid,
        all_to_all_indexes=data_dist['all_to_all_indexes'],
        maximal_distance=normalized_maximal_distance,
        registration_threshold=p_same_threshold,
        centroid_locations=aligned_centroid_locations,
        registration_approach=registration_approach,
        transform_data=False,
        verbose=False,
    )
    n_before_filter = _count_all_sessions(centroid_primary_map, n_sessions)

    # Step 2: post-filter — veto matched pairs with low spatial correlation
    from cellregpy import compute_spatial_correlation
    filtered_map = centroid_primary_map.copy()
    n_vetoed = 0
    for cluster_idx in range(filtered_map.shape[0]):
        sess_entries = filtered_map[cluster_idx, :]
        present_sessions = np.where(sess_entries > 0)[0]
        if len(present_sessions) < 2:
            continue
        # Check all pairwise spatial correlations within this cluster
        veto = False
        for ii in range(len(present_sessions)):
            if veto:
                break
            si = present_sessions[ii]
            ci = int(sess_entries[si]) - 1  # 0-indexed cell index
            for jj in range(ii + 1, len(present_sessions)):
                sj = present_sessions[jj]
                cj = int(sess_entries[sj]) - 1
                # Compute actual spatial correlation between the two footprints
                fp_i = aligned_fps[si][ci]
                fp_j = aligned_fps[sj][cj]
                sc = compute_spatial_correlation(fp_i, fp_j)
                if sc < spatial_corr_floor:
                    veto = True
                    break
        if veto:
            # Remove ALL multi-session links, keep only first present session
            # (effectively dissolves this cluster into singletons)
            for s_idx in present_sessions[1:]:
                filtered_map[cluster_idx, s_idx] = 0
            n_vetoed += 1

    print(f"  [Centroid-primary] {centroid_primary_map.shape[0]} clusters "
          f"({n_before_filter} in all sessions before filter)")
    print(f"  [Centroid-primary] Vetoed {n_vetoed} clusters with spatial corr < {spatial_corr_floor}")

    # --- Count cells in all sessions for each approach ---

    n_all_original  = _count_all_sessions(optimal_cell_to_index_map, n_sessions)
    n_all_consensus = _count_all_sessions(consensus_map, n_sessions)
    n_all_combined  = _count_all_sessions(combined_map, n_sessions)
    n_all_centprim  = _count_all_sessions(filtered_map, n_sessions)

    print(f"\n  --- Cells in ALL {n_sessions} sessions ---")
    print(f"  Original ({best_model_string}):        {n_all_original}")
    print(f"  Consensus (both models):               {n_all_consensus}")
    print(f"  Combined (p_cent × p_spat):            {n_all_combined}")
    print(f"  Centroid-primary (floor={spatial_corr_floor}): {n_all_centprim}")

    # --- Comparison figure: 4-panel projections ---
    def _make_projection(fp_list, cmap, session_idx):
        """Generate RGB projection for one session from a cell_to_index_map."""
        fps = fp_list[session_idx]
        if fps.ndim != 3:
            return np.zeros((100, 100, 3))
        h, w = fps.shape[1], fps.shape[2]
        n_sess = cmap.shape[1]

        c_idxs = cmap[:, session_idx].astype(int).copy()
        absent = c_idxs <= 0
        c_idxs -= 1
        c_idxs[absent] = -1

        # Normalize footprints
        norm = np.zeros_like(fps, dtype=float)
        for k in range(fps.shape[0]):
            fp = fps[k].astype(float)
            mx = float(np.max(fp))
            if mx > 0:
                fp[fp < 0.5 * mx] = 0
                norm[k] = fp / mx

        present_counts = np.sum(cmap > 0, axis=1)
        idx_all = np.where(present_counts == n_sess)[0]
        idxs_all = c_idxs[idx_all]; idxs_all = idxs_all[idxs_all >= 0]
        other_mask = (c_idxs >= 0) & (present_counts < n_sess)
        idxs_other = c_idxs[other_mask]; idxs_other = idxs_other[idxs_other >= 0]

        img = np.zeros((h, w, 3), dtype=float)
        s_all = np.sum(norm[idxs_all], axis=0) if len(idxs_all) > 0 else np.zeros((h, w))
        s_oth = np.sum(norm[idxs_other], axis=0) if len(idxs_other) > 0 else np.zeros((h, w))
        img[..., 0] = s_oth
        img[..., 1] = s_oth + s_all
        img[..., 2] = s_oth
        return np.clip(img, 0, 1)

    approaches = [
        (optimal_cell_to_index_map, f"Original\n({best_model_string})\n{n_all_original} in all"),
        (consensus_map,             f"Consensus\n(both models)\n{n_all_consensus} in all"),
        (combined_map,              f"Combined p_same\n(centroid × spatial)\n{n_all_combined} in all"),
        (filtered_map,              f"Centroid-primary\n(spatial floor={spatial_corr_floor})\n{n_all_centprim} in all"),
    ]

    fig, axes = plt.subplots(len(approaches), n_sessions,
                             figsize=(4 * n_sessions, 4 * len(approaches)))
    fig.suptitle('Stage 6 — Dual-Model Comparison', fontsize=16, fontweight='bold', y=0.98)

    for row, (cmap_row, label) in enumerate(approaches):
        for col in range(n_sessions):
            ax = axes[row, col] if len(approaches) > 1 else axes[col]
            proj = _make_projection(aligned_fps, cmap_row, col)
            ax.imshow(proj)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f'Session {col + 1}', fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_both(fig, os.path.join(fig_dir, "Stage 6 - dual model comparison"), show=SHOW_FIGURES)
    if not INLINE_PLOTS and not SHOW_FIGURES:
        plt.close(fig)
    print("  ✓ Stage 6 comparison figure done")

    # ================================================================== #
    #  Save results (MATLAB lines 662-693)
    # ================================================================== #
    print("\n--- Saving results ---")
    np.save(results_directory / 'cell_to_index_map.npy', optimal_cell_to_index_map)
    np.save(results_directory / 'cell_to_index_map_consensus.npy', consensus_map)
    np.save(results_directory / 'cell_to_index_map_combined.npy', combined_map)
    np.save(results_directory / 'cell_to_index_map_centroid_primary.npy', filtered_map)
    print(f"  Saved: {results_directory / 'cell_to_index_map.npy'}")
    print(f"  Saved: {results_directory / 'cell_to_index_map_consensus.npy'}")
    print(f"  Saved: {results_directory / 'cell_to_index_map_combined.npy'}")
    print(f"  Saved: {results_directory / 'cell_to_index_map_centroid_primary.npy'}")

    # ================================================================== #
    #  Summary
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Figures saved to: {figures_directory}")
    print(f"  Results saved to: {results_directory}")
    print(f"  Total cells registered: {optimal_cell_to_index_map.shape[0]}")
    print(f"  Best model: {best_model_string}")
    print(f"  Consensus cells:        {consensus_map.shape[0]}")
    print(f"  Combined cells:         {combined_map.shape[0]}")
    print(f"  Centroid-primary cells:  {filtered_map.shape[0]} ({n_all_centprim} in all sessions)")
    print("=" * 60)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == "__main__":
    import traceback
    try:
        run_pipeline(folder_path)
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"  PIPELINE ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
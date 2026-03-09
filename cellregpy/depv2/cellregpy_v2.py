"""
CellRegPy: Python port of batchRunCellReg MATLAB code.

Cross-session cell alignment using:
1. Mean image alignment (innovation over standard CellReg)
2. Probabilistic cell matching based on spatial footprints and centroids
3. Mouse data table generation for downstream analysis

Original MATLAB code by John Stout (SpellmanLab)
Python port: 2026-01-26

Key improvements over standard CellReg:
- Mean image alignment using multi-transform search with high-pass filtering
- Automatic detection of alignable sessions via correlation thresholding
- Builds integrated mouse_data and mouse_table structures
"""

# TODO: Stop using .mat files for spatial footprints. Just load straight from suite2p ops variable/stat variable.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import ndimage, stats, optimize
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
from functools import partial
import warnings
import os
import re
import traceback
import pickle
from datetime import datetime


# ============================================================================ #
#                           CONFIGURATION DATACLASS                            #
# ============================================================================ #
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CellRegConfig:
    """
    Configuration parameters for CellRegPy.
    Mirrors the parameter space from batchRunCellReg.m.
    """

    # Memory and display
    memory_efficient_run: bool = True
    figures_visibility: str = 'on'   # changed from 'off'

    # Pixel scaling
    microns_per_pixel: float = 2.0

    # Redundancy removal
    remove_redundancies: bool = True

    # Alignment correlation thresholds
    sufficient_correlation_centroids: float = 0.2
    sufficient_correlation_footprints: float = 0.3
    correlation_threshold: float = 0.65

    # Alignment modeling
    alignment_type: str = 'translations_and_rotations'
    use_parallel_processing: bool = True
    maximal_rotation: float = 30.0
    transformation_smoothness: float = 2.0

    # Probabilistic modeling
    maximal_distance: float = 14.0
    p_same_certainty_threshold: float = 0.95

    # Final registration
    registration_approach: str = 'Probabilistic'
    p_same_threshold: float = 0.5

    # Probabilistic model selection
    model_type: str = "auto"

    # Dual-model final registration
    dual_model: bool = False
    apply_spatial_floor_filter: bool = False
    spatial_corr_floor: float = 0.5

    # Figure saving
    save_figures: bool = True        # changed from False
    also_pdf: bool = True
    close_figures: bool = True

    # Debug / short runs
    test_run: bool = False
    test_run_type: str = 'test random alignment'

    # Mean image alignment parameters
    blur_hp: float = 12.0
    blur_lp: float = 5.0
    blur_bp1: float = 2.0
    blur_bp2: float = 12.0
    blur_reg: float = 2.0
    min_overlap_hard: float = 0.25
    gamma_overlap: float = 0.75
    z_thresh: float = 8.0
    min_area: int = 25
    alignable_threshold: float = 0.3

    # Alignment fallback options
    alignment_fallback_mode: str = 'two_stage'
    footprint_projection_threshold: float = 0.5
    footprint_filter_mode: str = 'highpass'
    footprint_outlier_mode: str = 'off'

    # Auto-simple mode
    auto_simple_on_high_similarity: bool = False
    auto_simple_raw_corr_threshold: float = 0.90
    auto_simple_aligned_corr_threshold: float = 0.95
    auto_simple_method: str = 'iou_hungarian'

    # Auto-flex mode
    auto_flex_on_high_peak: bool = True
    auto_flex_peak_threshold: float = 0.95
    auto_flex_maximal_distance_um_candidates: Tuple[float, ...] = (14.0, 7.0)
    auto_flex_disp_target_um: float = 2.0
    auto_flex_choose_best: bool = True

    # Simple IoU+Hungarian registration parameters
    simple_mask_threshold: float = 0.15   # changed from 0.20
    simple_iou_threshold: float = 0.10    # changed from 0.25
    simple_dist_threshold_um: float = 6.0
    simple_cost_beta: float = 0.25

    @property
    def normalized_maximal_distance(self) -> float:
        return self.maximal_distance / self.microns_per_pixel

# ============================================================================ #
#                          SUITE2P DATA LOADING                                #
# ============================================================================ #

def load_cellreg_mat(path: Path) -> Dict[str, np.ndarray]:
    """
    Load a CellReg.mat file from suite2p/plane0 folder.
    
    Args:
        path: Path to the CellReg.mat file
        
    Returns:
        Dictionary containing spatial footprints and metadata
    """
    data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return data


def load_fall_mat(plane0_path: Path) -> Dict[str, Any]:
    """
    Load Fall.mat from suite2p plane0 folder.
    
    Args:
        plane0_path: Path to plane0 folder containing Fall.mat
        
    Returns:
        Dictionary with F, Fneu, spks, stat, ops, iscell, etc.
    """
    fall_path = Path(plane0_path) / 'Fall.mat'
    if not fall_path.exists():
        raise FileNotFoundError(f"Fall.mat not found in {plane0_path}")
    
    data = loadmat(str(fall_path), squeeze_me=True, struct_as_record=False)
    
    result = {}
    for key in ['F', 'Fneu', 'spks', 'iscell']:
        if key in data:
            result[key] = np.array(data[key])
    
    # Handle ops structure
    if 'ops' in data:
        ops = data['ops']
        result['ops'] = {}
        # Get commonly used fields
        for field in ['meanImg', 'meanImgE', 'Ly', 'Lx', 'fs', 'nframes', 'xoff', 'yoff']:
            if hasattr(ops, field):
                result['ops'][field] = getattr(ops, field)
    
    # Handle stat (cell statistics)
    if 'stat' in data:
        result['stat'] = data['stat']
    
    return result


def get_mean_image(plane0_path: Path, apply_drift_correction: bool = True) -> np.ndarray:
    """
    Extract mean image from Fall.mat with optional drift correction.
    
    The mean image is computed from raw (non-motion-corrected) frames, while 
    spatial footprints are extracted from motion-corrected frames. To properly
    align mean images with footprints, we apply the average drift offset.
    
    Args:
        plane0_path: Path to plane0 folder
        apply_drift_correction: If True, apply drift correction using xoff/yoff
            from ops. This shifts the mean image to align with motion-corrected
            spatial footprints. Default True.
        
    Returns:
        Mean image as 2D numpy array (drift-corrected if requested)
    """
    from scipy.ndimage import shift
    
    fall = load_fall_mat(plane0_path)
    if 'ops' not in fall or 'meanImg' not in fall['ops']:
        raise ValueError(f"meanImg not found in {plane0_path}/Fall.mat")
    
    mean_img = np.array(fall['ops']['meanImg'])
    
    if apply_drift_correction:
        ops = fall['ops']
        xoff = np.array(ops.get('xoff', [0]))
        yoff = np.array(ops.get('yoff', [0]))
        
        # Mean shift to align raw mean image to motion-corrected reference
        dx = np.mean(xoff)
        dy = np.mean(yoff)
        
        if dx != 0 or dy != 0:
            # shift() uses [row, col] = [y, x] ordering
            mean_img = shift(mean_img, [dy, dx], mode='constant', cval=np.nan)
    
    return mean_img


def get_spatial_footprints(cellreg_path: Union[Path, str]) -> np.ndarray:
    """
    Load spatial footprints from CellReg.mat file.
    
    Handles multiple MATLAB formats:
    - 'footprint': Cell array of sparse matrices (from batchRunCellReg.m)
    - 'spatial_footprints': Standard CellReg output
    - 'A': Suite2p-style
    
    Args:
        cellreg_path: Path to CellReg.mat file
        
    Returns:
        3D array of shape (n_cells, height, width)
    """
    from scipy.sparse import issparse
    from scipy.io import loadmat
    
    data = loadmat(str(cellreg_path), squeeze_me=True, struct_as_record=False)
    
    # Get all non-system keys (exclude __header__, __version__, __globals__)
    user_keys = [k for k in data.keys() if not k.startswith('__')]
    
    # Look for known keys first (order matters: most specific first)
    footprints = None
    for key in ['footprint', 'footprints', 'spatial_footprints', 'A']:
        if key in data:
            footprints = data[key]
            print(f"  Loaded footprints from key '{key}'")
            break
    
    # MATLAB fallback: if no known key found, grab the FIRST variable (like load_footprint_data.m)
    if footprints is None and user_keys:
        first_key = user_keys[0]
        footprints = data[first_key]
        print(f"  Using first variable '{first_key}' (MATLAB-style fallback)")
    
    if footprints is None:
        print(f"DEBUG: Keys found in {Path(cellreg_path).name}: {user_keys}")
        raise ValueError(f"Could not find spatial footprints in {cellreg_path}")
    
    # Handle MATLAB cell array of sparse matrices
    # This is what batchRunCellReg produces via mat_to_sparse_cell
    if isinstance(footprints, np.ndarray) and footprints.dtype == object:
        # Cell array: each element is a sparse matrix for one cell
        n_cells = len(footprints)
        
        # Get shape from first element
        first = footprints.flat[0]
        if issparse(first):
            h, w = first.shape
        else:
            h, w = first.shape
            
        # Stack into 3D array
        result = np.zeros((n_cells, h, w), dtype=np.float32)
        for i, fp in enumerate(footprints.flat):
            if issparse(fp):
                result[i] = fp.toarray()
            else:
                result[i] = fp
        
        footprints = result
        
    elif issparse(footprints):
        # Handle single sparse matrix (n_pixels x n_cells)
        footprints = footprints.toarray()
        
    # Ensure 3D array
    footprints = np.array(footprints, dtype=np.float32)

    return footprints


def get_iscell(plane0_path: Path) -> np.ndarray:
    """
    Get iscell classification from Fall.mat.
    
    Args:
        plane0_path: Path to plane0 folder
        
    Returns:
        Boolean array of length n_rois indicating which are cells
    """
    fall = load_fall_mat(plane0_path)
    if 'iscell' in fall:
        iscell = fall['iscell']
        # iscell is typically (n_rois, 2) where first column is 0/1
        if iscell.ndim == 2:
            return iscell[:, 0].astype(bool)
        return iscell.astype(bool)
    raise ValueError(f"iscell not found in {plane0_path}/Fall.mat")


def list_session_folders(mouse_folder: Path) -> List[Path]:
    """
    List all session folders containing suite2p/plane0 within a mouse folder.
    
    Mirrors MATLAB: folder_names = listSubdirs(mouse_folder{mousei});
                    folder_names = folder_names(contains(folder_names, 'plane0'));
    
    Args:
        mouse_folder: Path to mouse data folder
        
    Returns:
        List of paths to plane0 folders
    """

    mouse_path = Path(mouse_folder)

    # Find Fall.mat that live specifically at suite2p/plane0/Fall.mat
    fall_files = mouse_path.rglob("suite2p/plane0/Fall.mat")
    plane0_folders = {p.parent for p in fall_files}  # p.parent is .../plane0

    return sorted(plane0_folders)

def suite2pToCellReg(fnames, mask_overlap: bool = True,
                     save_name: str = 'CellReg.mat'):
    """
    Convert suite2p results to a CellReg-compatible spatial-footprint .mat file.

    Port of ``sessreg.suite2pToCellReg`` — uses ``ravel_multi_index`` to build
    dense footprint matrices from suite2p ``stat.npy`` pixel coordinates.

    Args:
        fnames: A list of directories.  Each may be:
            - the root recording folder (containing ``suite2p/plane0/``),
            - the ``suite2p/`` folder itself, or
            - the ``plane0/`` (or any ``planeN/``) folder directly.
            A single string is also accepted.
        mask_overlap: If True (default), zero out pixels flagged as
            overlapping with neighbouring ROIs (``stat[i]['overlap']``).
        save_name: Filename for the saved ``.mat`` file
            (default ``'CellReg.mat'``).

    Returns:
        footprint: The last footprint array that was saved (n_cells, Ly, Lx).
    """
    from scipy.io import savemat as _savemat
    import os

    # Accept a single string
    if isinstance(fnames, (str, Path)):
        fnames = [fnames]

    footprint = None
    for fi in fnames:
        fi = str(fi)

        # Resolve to the plane folder -----------------------------------------------
        # If already pointing at a planeN folder, use it directly
        fiabs = os.path.abspath(fi)
        if os.path.basename(fiabs).startswith('plane'):
            plane_dir = fiabs
        elif os.path.isdir(os.path.join(fiabs, 'suite2p', 'plane0')):
            plane_dir = os.path.join(fiabs, 'suite2p', 'plane0')
        elif os.path.isdir(os.path.join(fiabs, 'plane0')):
            plane_dir = os.path.join(fiabs, 'plane0')
        else:
            # Try to find any planeN folder
            for sub in sorted(os.listdir(fiabs)):
                if sub.startswith('plane') and os.path.isdir(os.path.join(fiabs, sub)):
                    plane_dir = os.path.join(fiabs, sub)
                    break
            else:
                plane_dir = fiabs  # last resort

        # Load suite2p outputs -------------------------------------------------------
        stat = np.load(os.path.join(plane_dir, 'stat.npy'), allow_pickle=True)
        ops  = np.load(os.path.join(plane_dir, 'ops.npy'), allow_pickle=True).item()

        if mask_overlap:
            print("Removing cells with overlap")

        # Build dense footprints -----------------------------------------------------
        n_cells = len(stat)
        footprint = np.zeros((n_cells, ops['Lx'], ops['Ly']))

        for it_cell in range(n_cells):
            footprint_cell = np.zeros((footprint.shape[1], footprint.shape[2]))
            idx = np.ravel_multi_index(
                (stat[it_cell]['ypix'], stat[it_cell]['xpix']),
                dims=(footprint.shape[1], footprint.shape[2]),
            )

            if mask_overlap:
                overlap = stat[it_cell].get('overlap', None)
                if overlap is not None:
                    overlap = np.asarray(overlap).ravel().astype(bool)
                    if len(overlap) == len(idx):
                        idx = idx[~overlap]
                        lam = stat[it_cell]['lam'][~overlap]
                    else:
                        lam = stat[it_cell]['lam']
                else:
                    lam = stat[it_cell]['lam']
            else:
                lam = stat[it_cell]['lam']

            footprint_cell.flat[idx] = lam
            footprint[it_cell, :, :] = footprint_cell

        # Save -----------------------------------------------------------------------
        save_dir = plane_dir
        save_path = os.path.join(save_dir, save_name)
        print(f"Saving CellReg-compatible footprint array to {save_path}")
        _savemat(save_path, {'footprints': footprint})

    return footprint


def get_cellreg_files(plane0_folders: List[Path],
                      auto_generate: bool = True) -> List[Path]:
    """
    Find CellReg.mat files in plane folders.

    If ``auto_generate`` is True and a CellReg.mat is not found but
    ``stat.npy`` exists, ``suite2pToCellReg`` is called to build and save
    the footprints automatically.

    Args:
        plane0_folders: List of paths to plane folders.
        auto_generate: Build CellReg.mat from stat.npy when missing.

    Returns:
        List of paths to CellReg.mat files.
    """
    cellreg_files = []

    for folder in plane0_folders:
        folder_path = Path(folder)
        # Look for CellReg.mat (case-insensitive)
        found = None
        try:
            for f in folder_path.iterdir():
                if f.is_file() and 'cellreg' in f.name.lower() and f.suffix == '.mat':
                    found = f
                    break
        except OSError:
            pass

        if found is not None:
            cellreg_files.append(found)
        elif auto_generate and (folder_path / "stat.npy").exists():
            print(f"  CellReg.mat not found in {folder_path.name} – "
                  f"generating from stat.npy …")
            suite2pToCellReg(str(folder_path), mask_overlap=True,
                             save_name='CellReg.mat')
            mat_path = folder_path / 'CellReg.mat'
            cellreg_files.append(mat_path)

    return cellreg_files


# ============================================================================ #
#                         HELPER FUNCTIONS                                     #
# ============================================================================ #

def ensure_valid_field_name(name: str) -> str:
    """
    Convert string to valid Python/MATLAB field name.
    
    Mirrors MATLAB ensureValidFieldName().
    """
    # Replace invalid characters with underscores
    name = re.sub(r'[^\w]', '_', str(name))
    # Ensure doesn't start with number
    if name and name[0].isdigit():
        name = '_' + name
    return name



def _truncate_field_name(name: str, max_len: int = 63) -> str:
    """Truncate a (sanitized) field name to <= max_len chars with a stable hash suffix."""
    name = str(name)
    if len(name) <= max_len:
        return name
    import hashlib
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    keep = max_len - 9  # "_" + 8 hex
    return f"{name[:keep]}_{h}"


def sanitize_for_mat(obj: Any) -> Any:
    """Recursively sanitize nested dict keys + problematic values for scipy.io.savemat."""
    if obj is None:
        return np.nan
    # pathlib Paths -> strings (savemat can't handle Path objects)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        used = set()
        for k, v in obj.items():
            kk = _truncate_field_name(ensure_valid_field_name(k))
            # ensure unique
            base = kk
            c = 1
            while kk in used:
                c += 1
                kk = _truncate_field_name(f"{base}_{c}")
            used.add(kk)
            out[kk] = sanitize_for_mat(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_mat(x) for x in obj]
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return np.array([sanitize_for_mat(x) for x in obj], dtype=object)
        return obj
    return obj


def get_session_unix_time(session_root: Union[str, Path]) -> Optional[int]:
    """Extract Unix time from Experiment.xml (matches MATLAB metadata.Date.uTimeAttribute)."""
    session_root = Path(session_root)
    exp = session_root / "Experiment.xml"
    if not exp.exists():
        # light fallback search
        try:
            exp = next(session_root.rglob("Experiment.xml"))
        except StopIteration:
            return None
        except Exception:
            return None
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(exp))
        root = tree.getroot()
        # look for a tag named 'Date' anywhere
        date_el = None
        for el in root.iter():
            if el.tag.lower().endswith("date"):
                date_el = el
                break
        if date_el is None:
            return None
        # common attribute names
        for key in ("uTime", "utime", "uTimeAttribute", "utimeattribute"):
            if key in date_el.attrib:
                val = date_el.attrib[key]
                try:
                    return int(float(val))
                except Exception:
                    return None
        return None
    except Exception:
        return None

def empty_cell_erase(cell_list: List) -> Tuple[List, List[int]]:
    """
    Remove empty elements from list and return indices of removed elements.
    
    Mirrors MATLAB emptyCellErase().
    """
    result = []
    removed_indices = []
    
    for i, item in enumerate(cell_list):
        if item is None or (hasattr(item, '__len__') and len(item) == 0):
            removed_indices.append(i)
        else:
            result.append(item)
    
    return result, removed_indices


def as_num_row(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is a 1D row vector.
    
    Mirrors MATLAB asNumRow().
    """
    arr = np.atleast_1d(np.array(arr).flatten())
    return arr

def _norm01(img):
    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img)
    img = (img - lo) / (hi - lo)
    return np.clip(img, 0, 1)

def _rgb_overlay(fixed, other):
    f = _norm01(fixed)
    o = _norm01(other)
    rgb = np.zeros((*f.shape, 3), dtype=float)
    rgb[..., 0] = o        # red = moving/registered
    rgb[..., 1] = f        # green = fixed
    rgb[..., 2] = f        # blue  = fixed  -> cyan
    return rgb


def compute_registered_pair_displacements(cell_to_index_map: np.ndarray,
                                         centroid_locations: List[np.ndarray],
                                         ref_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dx, dy (in pixels) for registered pairs (ref vs each other session).

    dx = x_other - x_ref, dy = y_other - y_ref, pooled across all non-ref sessions.
    Only rows where BOTH sessions have a non-zero entry are used.
    """
    cmap = np.asarray(cell_to_index_map, dtype=int)
    n_rows, n_sessions = cmap.shape if cmap.ndim == 2 else (0, 0)
    if n_rows == 0 or n_sessions == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    dx_all = []
    dy_all = []
    for s in range(n_sessions):
        if s == ref_idx:
            continue
        mask = (cmap[:, ref_idx] > 0) & (cmap[:, s] > 0)
        if not np.any(mask):
            continue
        ref_ids = cmap[mask, ref_idx].astype(int) - 1
        oth_ids = cmap[mask, s].astype(int) - 1
        ref_c = np.asarray(centroid_locations[ref_idx])[ref_ids]
        oth_c = np.asarray(centroid_locations[s])[oth_ids]
        dx_all.append((oth_c[:, 0] - ref_c[:, 0]).astype(float))
        dy_all.append((oth_c[:, 1] - ref_c[:, 1]).astype(float))

    if len(dx_all) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(dx_all), np.concatenate(dy_all)


def displacement_quality(dx_px: np.ndarray,
                         dy_px: np.ndarray,
                         microns_per_pixel: float,
                         target_um: float = 2.0) -> Dict[str, float]:
    """Summarize 'goodness' of a displacement cloud.

    Returns metrics in microns: match_count, median_r_um, frac_within_target, mean_dx_um, mean_dy_um.
    """
    dx_px = np.asarray(dx_px, dtype=float).ravel()
    dy_px = np.asarray(dy_px, dtype=float).ravel()
    if dx_px.size == 0 or dy_px.size == 0:
        return dict(match_count=0.0, median_r_um=float('nan'), frac_within_target=float('nan'),
                    mean_dx_um=float('nan'), mean_dy_um=float('nan'))

    r_px = np.sqrt(dx_px**2 + dy_px**2)
    r_um = r_px * float(microns_per_pixel)

    target_px = float(target_um) / float(microns_per_pixel)
    frac = float(np.mean(r_px <= target_px)) if r_px.size else float('nan')

    return dict(
        match_count=float(r_px.size),
        median_r_um=float(np.nanmedian(r_um)),
        frac_within_target=float(frac),
        mean_dx_um=float(np.nanmean(dx_px) * float(microns_per_pixel)),
        mean_dy_um=float(np.nanmean(dy_px) * float(microns_per_pixel)),
    )


def neighbor_displacement_quality(data_dist: Dict[str, Any],
                                  microns_per_pixel: float,
                                  target_um: float = 2.0) -> Dict[str, float]:
    """Quality metrics computed from the *neighbor-pair* displacement vectors (Stage 3 inputs)."""
    dx = np.asarray(data_dist.get('neighbors_x_displacements', []), dtype=float).ravel()
    dy = np.asarray(data_dist.get('neighbors_y_displacements', []), dtype=float).ravel()
    return displacement_quality(dx, dy, microns_per_pixel, target_um=target_um)
# ============================================================================ #
#                      MEAN IMAGE ALIGNMENT                                    #
# ============================================================================ #

class MeanImageAligner:
    """
    Mean image alignment engine.
    
    Ports mean_img_alignment.m with multi-transform search,
    high-pass/low-pass/band-pass filtering, and outlier suppression.
    
    This is the key innovation over standard CellReg.
    """
    
    def __init__(self, config: Optional[CellRegConfig] = None):
        self.config = config or CellRegConfig()
        
        # Transform methods to try
        self.methods = ['identity', 'translation', 'rigid', 'similarity', 'affine']
    
    def align(self, 
              fixed_img: np.ndarray, 
              moving_img: np.ndarray,
              filter_mode: str = 'highpass',
              outlier_mode: str = 'auto',
              plot_fig: bool = False) -> Tuple[np.ndarray, str, float, Any, str, bool]:
        """
        Align moving image to fixed image using best transform.
        
        Mirrors MATLAB mean_img_alignment().
        
        Args:
            fixed_img: Reference image
            moving_img: Image to align
            filter_mode: 'highpass', 'lowpass', 'bandpass', or 'auto'
            outlier_mode: 'on', 'off', or 'auto'
            plot_fig: Whether to show diagnostic plot
            
        Returns:
            Tuple of (registered_image, best_method, best_peak, best_transform, 
                      best_filter, best_outliers)
        """
        from skimage import transform as sktransform
        from skimage.registration import phase_cross_correlation
        from skimage.metrics import normalized_mutual_information
        from scipy.ndimage import affine_transform
        
        cfg = self.config
        
        # Determine filter modes to try
        do_hp, do_lp, do_bp = self._parse_filter_mode(filter_mode)
        
        # Determine outlier modes to try  
        if outlier_mode == 'auto':
            outlier_opts = [False, True]
        elif outlier_mode == 'on':
            outlier_opts = [True]
        else:
            outlier_opts = [False]
        
        # Prepare images
        fixed = self._to_float(fixed_img)
        moving = self._to_float(moving_img)
        
        # Evaluate identity first to establish baseline
        identity_tform = sktransform.AffineTransform()
        pc_hp_id, pc_lp_id, pc_bp_id, _, ov_frac_id = self._eval_one_combo(
            fixed=fixed, moving=moving, method='identity', outlier_mode=False,
            do_hp=do_hp, do_lp=do_lp, do_bp=do_bp
        )
        
        # Use identity as the baseline
        best_peak = -np.inf
        # Set to identity baseline if metrics exist
        if do_hp: best_peak = pc_hp_id
        elif do_lp: best_peak = pc_lp_id
        elif do_bp: best_peak = pc_bp_id
            
        best_method = 'identity'
        best_filter = 'highpass' if do_hp else ('lowpass' if do_lp else 'bandpass')
        best_outliers = False
        best_tform = identity_tform
        
        # Add margin for accepting a new transform
        margin = 0.01 
        
        # Collect all scores for the scoreboard
        all_results = [{'method': 'identity', 'outliers': False, 
                        'hp': pc_hp_id, 'lp': pc_lp_id, 'bp': pc_bp_id,
                        'tform': identity_tform, 'params': 'n/a'}]
        
        for use_out in outlier_opts:
            for method in self.methods:
                if method == 'identity':
                    continue
                pc_hp, pc_lp, pc_bp, tform, ov_frac = self._eval_one_combo(
                    fixed=fixed, moving=moving, method=method, outlier_mode=use_out,
                    do_hp=do_hp, do_lp=do_lp, do_bp=do_bp
                )
                
                # Extract transform parameters for logging
                tform_desc = 'n/a'
                if tform is not None:
                    try:
                        p = tform.params
                        tx, ty = float(p[0, 2]), float(p[1, 2])
                        rot = float(np.degrees(np.arctan2(p[1, 0], p[0, 0])))
                        sx = float(np.sqrt(p[0, 0]**2 + p[1, 0]**2))
                        sy = float(np.sqrt(p[0, 1]**2 + p[1, 1]**2))
                        shear = float(np.degrees(np.arctan2(-p[0, 1], p[1, 1]))) - 90 + rot
                        tform_desc = (f"tx={tx:.1f}, ty={ty:.1f}, rot={rot:.2f}°, "
                                      f"sx={sx:.4f}, sy={sy:.4f}")
                        if method == 'affine':
                            tform_desc += f", shear={shear:.2f}°"
                    except Exception:
                        tform_desc = 'error reading params'
                
                # Log per-method results
                scores_str = []
                if do_hp: scores_str.append(f"hp={pc_hp:.4f}")
                if do_lp: scores_str.append(f"lp={pc_lp:.4f}")
                if do_bp: scores_str.append(f"bp={pc_bp:.4f}")
                print(f"  {method:12s} (out={use_out}): {', '.join(scores_str)} | {tform_desc}")
                
                all_results.append({'method': method, 'outliers': use_out,
                                    'hp': pc_hp, 'lp': pc_lp, 'bp': pc_bp,
                                    'tform': tform, 'params': tform_desc})
                
                # Check each filter type against the best + margin
                if do_hp and pc_hp > (best_peak + margin):
                    best_peak = pc_hp
                    best_filter = 'highpass'
                    best_outliers = use_out
                    best_method = method
                    best_tform = tform
                    
                if do_lp and pc_lp > (best_peak + margin):
                    best_peak = pc_lp
                    best_filter = 'lowpass'
                    best_outliers = use_out
                    best_method = method
                    best_tform = tform
                    
                if do_bp and pc_bp > (best_peak + margin):
                    best_peak = pc_bp
                    best_filter = 'bandpass'
                    best_outliers = use_out
                    best_method = method
                    best_tform = tform
        
        # --- Scoreboard summary ---
        identity_peak = pc_hp_id if do_hp else (pc_lp_id if do_lp else pc_bp_id)
        print(f"\n  ╔══ Alignment Scoreboard ════════════════════════════════════")
        print(f"  ║ {'Method':<15s} {'Out':>3s}  {'HP':>8s}  {'LP':>8s}  {'BP':>8s}  {'Δ vs id':>8s}")
        print(f"  ╟─────────────────────────────────────────────────────────────")
        for r in all_results:
            primary = r['hp'] if do_hp else (r['lp'] if do_lp else r['bp'])
            delta = primary - identity_peak if np.isfinite(primary) else float('nan')
            marker = ' ★' if r['method'] == best_method and r['outliers'] == best_outliers else '  '
            hp_s = f"{r['hp']:.4f}" if np.isfinite(r['hp']) else '  -inf'
            lp_s = f"{r['lp']:.4f}" if np.isfinite(r['lp']) else '  -inf'
            bp_s = f"{r['bp']:.4f}" if np.isfinite(r['bp']) else '  -inf'
            d_s = f"{delta:+.4f}" if np.isfinite(delta) else '    n/a'
            out_s = 'Y' if r['outliers'] else 'N'
            print(f"  ║{marker}{r['method']:<13s}   {out_s}  {hp_s}  {lp_s}  {bp_s}  {d_s}")
        print(f"  ╚══════════════════════════════════════════════════════════════")
        print(f"  Winner: {best_method} ({best_filter}), score={best_peak:.4f}")

        # --- Fallback: if no improvement found, try all filter modes ---
        if best_method == 'identity' and not (do_hp and do_lp and do_bp):
            print("  No improvement with initial filter mode, trying all filter modes...")
            # Re-evaluate identity with all filters for baseline
            pc_hp_all, pc_lp_all, pc_bp_all, _, _ = self._eval_one_combo(
                fixed=fixed, moving=moving, method='identity', outlier_mode=False,
                do_hp=True, do_lp=True, do_bp=True
            )
            # Track best per-filter-type baselines separately
            fb_baselines = {'highpass': pc_hp_all, 'lowpass': pc_lp_all, 'bandpass': pc_bp_all}
            best_peak_fb = max(
                pc_hp_all if np.isfinite(pc_hp_all) else -np.inf,
                pc_lp_all if np.isfinite(pc_lp_all) else -np.inf,
                pc_bp_all if np.isfinite(pc_bp_all) else -np.inf,
            )
            best_peak = best_peak_fb

            for use_out in outlier_opts:
                for method in self.methods:
                    if method == 'identity':
                        continue
                    pc_hp_fb, pc_lp_fb, pc_bp_fb, tform_fb, ov_fb = self._eval_one_combo(
                        fixed=fixed, moving=moving, method=method, outlier_mode=use_out,
                        do_hp=True, do_lp=True, do_bp=True
                    )
                    if pc_hp_fb > (best_peak + margin):
                        best_peak = pc_hp_fb
                        best_filter = 'highpass'
                        best_outliers = use_out
                        best_method = method
                        best_tform = tform_fb
                    if pc_lp_fb > (best_peak + margin):
                        best_peak = pc_lp_fb
                        best_filter = 'lowpass'
                        best_outliers = use_out
                        best_method = method
                        best_tform = tform_fb
                    if pc_bp_fb > (best_peak + margin):
                        best_peak = pc_bp_fb
                        best_filter = 'bandpass'
                        best_outliers = use_out
                        best_method = method
                        best_tform = tform_fb

            if best_method != 'identity':
                print(f"  Fallback found improvement: {best_method} ({best_filter}), "
                      f"score={best_peak:.4f}")

        # Apply the winning transform
        if best_method != 'identity' and best_peak > -np.inf and best_tform is not None:
            moving_warp = moving.copy()
            if best_outliers:
                moving_warp, _ = self._suppress_outliers(moving_warp)
            
            registered = self._apply_transform(moving_warp, best_tform, fixed.shape)
        else:
            registered = moving.copy()
            best_tform = identity_tform  # Ensure it is returned on fallback

        # --- Post-selection validation gate ---
        # Independent check: raw-image Pearson correlation must improve
        if best_method != 'identity':
            reg_valid = np.isfinite(registered) & np.isfinite(fixed)
            mov_valid = np.isfinite(moving) & np.isfinite(fixed)

            if reg_valid.sum() > 100 and mov_valid.sum() > 100:
                corr_aligned = np.corrcoef(
                    fixed[reg_valid].ravel(), registered[reg_valid].ravel())[0, 1]
                corr_unaligned = np.corrcoef(
                    fixed[mov_valid].ravel(), moving[mov_valid].ravel())[0, 1]

                if not np.isfinite(corr_aligned) or corr_aligned <= corr_unaligned:
                    print(f"  \u26a0 Alignment REJECTED by validation gate: "
                          f"raw corr aligned={corr_aligned:.4f} "
                          f"<= unaligned={corr_unaligned:.4f}")
                    registered = moving.copy()
                    best_method = 'identity'
                    best_tform = identity_tform
                else:
                    print(f"  \u2713 Alignment ACCEPTED: raw corr "
                          f"aligned={corr_aligned:.4f} > "
                          f"unaligned={corr_unaligned:.4f}")
            else:
                print(f"  \u26a0 Alignment REJECTED: insufficient valid pixels "
                      f"for validation")
                registered = moving.copy()
                best_method = 'identity'
                best_tform = identity_tform

        # ---- Visual check ----
        if plot_fig:
            self.plot_alignment_result(fixed, moving, registered, best_method, best_filter, best_peak)
            
        return registered, best_method, best_peak, best_tform, best_filter, best_outliers

    def plot_image_pair(self, fixed: np.ndarray, moving: np.ndarray, title: str = "Image Pair"):
        """Helper to plain plot two images side-by-side."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Fixed
        vmin, vmax = np.nanpercentile(fixed, [1, 99])
        ax[0].imshow(fixed, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0].set_title("Fixed Image")
        ax[0].axis('off')
        
        # Moving
        vmin, vmax = np.nanpercentile(moving, [1, 99])
        ax[1].imshow(moving, cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1].set_title("Moving Image")
        ax[1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_alignment_result(self, fixed, moving, registered, method, filter_type, score):
        """Helper to plot alignment results (overlay + side-by-side)."""
        import matplotlib.pyplot as plt
        
        # Prepare for overlay (red-cyan)
        def norm(img):
            img = np.nan_to_num(img)
            lo, hi = np.percentile(img, [1, 99])
            if hi <= lo: return np.zeros_like(img)
            return np.clip((img - lo) / (hi - lo), 0, 1)

        f_norm = norm(fixed)
        r_norm = norm(registered)
        
        rgb = np.zeros((*fixed.shape, 3))
        rgb[..., 0] = r_norm # Red = Registered (Moving)
        rgb[..., 1] = f_norm # Green = Fixed
        rgb[..., 2] = f_norm # Blue = Fixed -> Cyan
        
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        
        ax[0].imshow(rgb)
        ax[0].set_title("Overlay (Cyan=Fixed, Red=Reg)")
        ax[0].axis('off')
        
        vmin, vmax = np.nanpercentile(fixed, [1, 99])
        ax[1].imshow(fixed, cmap='gray', vmin=vmin, vmax=vmax)
        ax[1].set_title("Fixed")
        ax[1].axis('off')
        
        vmin, vmax = np.nanpercentile(moving, [1, 99])
        ax[2].imshow(moving, cmap='gray', vmin=vmin, vmax=vmax)
        ax[2].set_title("Moving (Original)")
        ax[2].axis('off')
        
        vmin, vmax = np.nanpercentile(registered, [1, 99])
        ax[3].imshow(registered, cmap='gray', vmin=vmin, vmax=vmax)
        ax[3].set_title(f"Registered\n{method} | {filter_type} | r={score:.3f}")
        ax[3].axis('off')
        
        plt.tight_layout()
        plt.show()
    def _parse_filter_mode(self, mode: str) -> Tuple[bool, bool, bool]:
        """Parse filter mode string into boolean flags."""
        mode = mode.lower()
        if mode == 'auto':
            return True, True, True
        elif mode == 'auto_nolp':
            return True, False, True
        elif mode in ('highpass', 'hp'):
            return True, False, False
        elif mode in ('lowpass', 'lp'):
            return False, True, False
        elif mode in ('bandpass', 'bp'):
            return False, False, True
        else:
            raise ValueError(f"Unknown filter_mode: {mode}")    
    
    def _to_float(self, img: np.ndarray) -> np.ndarray:
        """MATLAB im2single equivalent.

        - Unsigned integers are scaled to [0, 1] using dtype max.
        - Signed integers are scaled to [-1, 1] using the largest magnitude of dtype range.
        - Float arrays are cast to float32 without rescaling.
        """
        if img is None:
            raise ValueError("img is None")

        arr = np.asarray(img)

        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            denom = float(max(abs(info.min), info.max))
            if denom == 0:
                return arr.astype(np.float32)
            return (arr.astype(np.float32) / denom)

        return arr.astype(np.float32, copy=False)


    def _suppress_outliers(self, 
                          img: np.ndarray, 
                          method: str = 'inpaint') -> Tuple[np.ndarray, np.ndarray]:
        """
        Suppress bright outliers in image.
        
        Mirrors MATLAB suppress_outliers() helper.
        """
        from scipy.ndimage import median_filter
        from scipy.stats import median_abs_deviation
        
        cfg = self.config
        
        # Handle NaN/inf values before filtering - replace with median of valid pixels
        img = img.copy()
        nan_mask = ~np.isfinite(img)
        if nan_mask.any():
            valid_vals = img[~nan_mask]
            if valid_vals.size > 0:
                fill_val = np.median(valid_vals)
            else:
                fill_val = 0.0
            img[nan_mask] = fill_val
        
        # High-pass to find outliers
        hp_img = img - gaussian_filter(img, cfg.blur_hp)
        
        # Robust z-score
        med_hp = np.median(hp_img)
        mad_hp = median_abs_deviation(hp_img.flatten(), scale='normal')
        rob_std = 1.4826 * mad_hp + 1e-10
        z_hp = (hp_img - med_hp) / rob_std
        
        # Create mask of outliers (bright only)
        threshold = max(cfg.z_thresh, np.percentile(z_hp, 99.9))
        mask = z_hp > threshold
        
        # Remove small regions (MATLAB: bwareaopen + imfill + imdilate)
        if cfg.min_area > 1:
            labeled, nlab = ndimage.label(mask)
            if nlab > 0:
                sizes = np.bincount(labeled.ravel())
                keep = sizes >= int(cfg.min_area)
                keep[0] = False  # background
                mask = keep[labeled]

        mask = ndimage.binary_fill_holes(mask)

        # MATLAB: imdilate(mask, strel('disk',1))
        # 1 px radius disk
        rr = 1
        ensure_dilation = True
        if ensure_dilation:
            # simple 3x3 cross or box
            mask = ndimage.binary_dilation(mask, structure=ndimage.generate_binary_structure(2, 1))

        # Apply correction
        if method == 'inpaint':
            # Simple inpainting: replace with local median
            img_out = img.copy()
            img_out[mask] = median_filter(img, size=5)[mask]
        else:  # clip
            valid = ~mask & np.isfinite(img)
            if valid.any():
                cap = np.percentile(img[valid], 99.8)
            else:
                cap = img.max()
            img_out = img.copy()
            img_out[mask] = cap
        
        return img_out, mask
    
    def _eval_one_combo(self,
                       fixed: np.ndarray,
                       moving: np.ndarray,
                       method: str,
                       outlier_mode: bool = False,
                       do_hp: bool = True,
                       do_lp: bool = False,
                       do_bp: bool = False) -> Tuple[float, float, float, Any, float]:
        """
        Evaluate one (transform, outlier) combination.
        
        Returns correlation scores for each filter type.
        """
        from skimage import transform as sktransform
        from skimage.registration import phase_cross_correlation
        from skimage.metrics import normalized_mutual_information
        from scipy.optimize import minimize
        
        cfg = self.config
        
        # Prepare masks for NaN values
        nan_mask_f = ~np.isfinite(fixed)
        nan_mask_m = ~np.isfinite(moving)
        
        # Determine strict valid mask (finite values)
        # In MATLAB, operations propagate NaNs, and then they are masked out by isfinite().
        # To mimic this in Python without erroring, we replace NaNs with 0 for filtering,
        # but WE MUST remember to exclude these regions from the final calculation.
        
        fixed_filled = fixed.copy()
        fixed_filled[nan_mask_f] = np.nanmedian(fixed) if np.any(~nan_mask_f) else 0
        
        moving_filled = moving.copy()
        moving_filled[nan_mask_m] = np.nanmedian(moving) if np.any(~nan_mask_m) else 0
        
        # Apply outlier suppression if requested
        mask_f = np.zeros_like(fixed, dtype=bool)
        mask_m = np.zeros_like(moving, dtype=bool)
        
        if outlier_mode:
            # _suppress_outliers handles NaNs internally now
            fixed_filled, mask_f = self._suppress_outliers(fixed) # This returns filled image
            moving_filled, mask_m = self._suppress_outliers(moving) # This returns filled image
            # Update NaN masks from original images which might have changed if suppression was applied? 
            # Actually _suppress_outliers returns INPAINTED image. 
            # We should trust its output for the content, but we effectively 'lost' the NaN info?
            # MATLAB's suppress_outliers returns infilled image too.
            pass
        
        # Pre-process with mild blur
        # We use the filled images for warping/filtering to avoid NaN propagation destroying the whole image
        fixed_f = gaussian_filter(fixed_filled, cfg.blur_reg)
        moving_f = gaussian_filter(moving_filled, cfg.blur_reg)
        
        # IMPORTANT: We must combine the original NaN mask with the outlier mask
        # MATLAB: baseMask = isfinite(movTmp) & isfinite(fixedF) & ~maskF & ~maskMov;
        # Since we filled NaNs, isfinite(fixed_f) will be all true. 
        # So we must explicitly add nan_mask_f to the exclusion list.
        mask_f = mask_f | nan_mask_f
        mask_m = mask_m | nan_mask_m

        def warp_img(img, tf, *, order=1, cval=np.nan):
            # preserve_range avoids skimage re-scaling intensities (MATLAB-like)
            return sktransform.warp(
                img,
                tf.inverse,
                output_shape=fixed_f.shape,
                mode="constant",
                cval=cval,
                order=order,
                preserve_range=True,
            )

        def warp_mask(mask, tf):
            # nearest-neighbor warp for masks (MATLAB 'nearest')
            w = sktransform.warp(
                mask.astype(np.uint8),
                tf.inverse,
                output_shape=fixed_f.shape,
                mode="constant",
                cval=0,
                order=0,
                preserve_range=True,
            )
            return w > 0.5

        # Identity early exit
        if method == 'identity':
            tform = sktransform.AffineTransform()
            overlap_frac = 1.0
            pc_hp, pc_lp, pc_bp = self._compute_scores(fixed_f, moving_f, mask_f, mask_m, 
                                                      do_hp, do_lp, do_bp, 1.0)
            return pc_hp, pc_lp, pc_bp, tform, overlap_frac

        # Initialize scores
        pc_hp, pc_lp, pc_bp = -np.inf, -np.inf, -np.inf
        tform = None
        overlap_frac = 0.0
        
        try:
            # Coarse translation seed using phase correlation
            # We use high-pass seed for coarse alignment to match MATLAB
            fix_co = gaussian_filter(fixed_f, 2) - gaussian_filter(fixed_f, 12)
            mov_co = gaussian_filter(moving_f, 2) - gaussian_filter(moving_f, 12)

            # affine transformation
            shift, _, _ = phase_cross_correlation(fix_co, mov_co, upsample_factor=1)
            t0 = sktransform.AffineTransform(translation=(shift[1], shift[0]))
            
            if method == 'translation':
                tform = t0
            else:
                # Iterative optimization for rotation/scale/etc.
                # Objective function: 1 - HP Energy Correlation
                def objective(params):
                    # --- Enforce parameter bounds to prevent destructive warping ---
                    max_trans = max(fixed_f.shape) * 0.3  # max 30% of image size
                    if abs(params[0]) > max_trans or abs(params[1]) > max_trans:
                        return 1.0
                    if abs(params[2]) > cfg.maximal_rotation:
                        return 1.0
                    if method == 'similarity':
                        if params[3] < 0.9 or params[3] > 1.1:
                            return 1.0
                    elif method == 'affine':
                        if params[3] < 0.9 or params[3] > 1.1 or params[4] < 0.9 or params[4] > 1.1:
                            return 1.0
                        if abs(params[5]) > 5.0:  # max 5 degrees shear
                            return 1.0

                    if method == 'rigid':
                        # params: [tx, ty, rot_deg]
                        curr_t = sktransform.EuclideanTransform(translation=(params[0], params[1]), 
                                                               rotation=np.deg2rad(params[2]))
                    elif method == 'similarity':
                        # params: [tx, ty, rot_deg, scale]
                        s = max(1e-6, params[3])
                        curr_t = sktransform.SimilarityTransform(
                            translation=(params[0], params[1]),
                            rotation=np.deg2rad(params[2]),
                            scale=s
                        )
                    elif method == 'affine':
                        # params: [tx, ty, rot_deg, scale_x, scale_y, shear_deg]
                        sx, sy = params[3], params[4]
                        if sx <= 0 or sy <= 0:
                            return 1.0  # invalid
                        curr_t = sktransform.AffineTransform(
                            translation=(params[0], params[1]),
                            rotation=np.deg2rad(params[2]),
                            scale=(sx, sy),
                            shear=np.deg2rad(params[5]),
                        )

                    # Warp and check overlap
                    warped = warp_img(moving_f, curr_t, order=1, cval=np.nan)
                    fixed_mask = (fixed_f > 0)
                    mov_mask0 = (moving_f > 0)
                    mov_mask_w = warp_mask(mov_mask0, curr_t)
                    valid = mov_mask_w & fixed_mask

                    # Outlier/NaN handling
                    # mask_f/mask_m now contain NaNs + outliers (if enabled)
                    # We must always exclude them to match MATLAB's isfinite() logic
                    mask_m_w = warp_mask(mask_m, curr_t)
                    base_mask = valid & (~mask_f) & (~mask_m_w)

                    # Also exclude warped-out-of-bounds NaNs
                    warped_nans = ~np.isfinite(warped)
                    if warped_nans.any():
                        base_mask = base_mask & (~warped_nans)

                    # hard overlap gate (MATLAB-like)
                    if base_mask.sum() < base_mask.size * cfg.min_overlap_hard:
                        return 1.0

                    # Compute Mutual Information score (negative for minimization)
                    # We compute MI only on the valid overlapping region to match registration behavior
                    # Use 50 bins to approximate MATLAB's Mattes MI default
                    
                    # Convert to consistent range for binning
                    f_valid = fixed_f[base_mask]
                    m_valid = warped[base_mask]

                    if f_valid.size < 100:
                        return 1.0
                    
                    # Double-check no NaN values remain
                    if not (np.all(np.isfinite(f_valid)) and np.all(np.isfinite(m_valid))):
                        return 1.0

                    # NMI (your choice) — higher is better, so negate for minimization
                    score = normalized_mutual_information(f_valid, m_valid, bins=50)
                    # Guard against NaN scores breaking optimizer convergence
                    if not np.isfinite(score):
                        return 1.0
                    return -score

                # Set initial params and bounds
                if method == 'rigid':
                    initial_params = [shift[1], shift[0], 0.0]
                elif method == 'similarity':
                    initial_params = [shift[1], shift[0], 0.0, 1.0]
                elif method == 'affine':
                    initial_params = [shift[1], shift[0], 0.0, 1.0, 1.0, 0.0]
                
                res = minimize(objective, initial_params, method='Powell', tol=1e-3)
                
                # Build final transform
                if method == 'rigid':
                    tform = sktransform.EuclideanTransform(translation=(res.x[0], res.x[1]), 
                                                          rotation=np.deg2rad(res.x[2]))
                elif method == 'similarity':
                    tform = sktransform.SimilarityTransform(translation=(res.x[0], res.x[1]),
                                                           rotation=np.deg2rad(res.x[2]),
                                                           scale=res.x[3])
                elif method == 'affine':
                    tform = sktransform.AffineTransform(
                        translation=(res.x[0], res.x[1]),
                        rotation=np.deg2rad(res.x[2]),
                        scale=(res.x[3], res.x[4]),
                        shear=np.deg2rad(res.x[5]))
                else:
                    # should never hit if method set is rigid/similarity/affine here
                    tform = sktransform.AffineTransform(translation=(res.x[0], res.x[1]))

            # Warp and check overlap
            mov_warped = warp_img(moving_f, tform, order=1, cval=np.nan)
            valid_mask = np.isfinite(mov_warped) & np.isfinite(fixed_f)

            
            # Always apply masking (NaN + Outliers)
            mask_m_w = warp_mask(mask_m, tform)
            base_mask = valid_mask & (~mask_f) & (~mask_m_w)

            overlap_frac = base_mask.sum() / base_mask.size

            # MATLAB-like fallback: if a fancy transform kills overlap, try coarse translation seed
            if overlap_frac < cfg.min_overlap_hard and method not in ('identity', 'translation'):
                mov_warped0 = warp_img(moving_f, t0, order=1, cval=np.nan)
                valid0 = np.isfinite(mov_warped0) & np.isfinite(fixed_f)

                # Always apply masking (NaN + Outliers)
                mask_m0 = warp_mask(mask_m, t0)
                base0 = valid0 & (~mask_f) & (~mask_m0)

                ov0 = base0.sum() / base0.size

                if ov0 >= cfg.min_overlap_hard:
                    tform = t0
                    mov_warped = mov_warped0
                    overlap_frac = ov0
                else:
                    return -np.inf, -np.inf, -np.inf, tform, overlap_frac

                
            pc_hp, pc_lp, pc_bp = self._compute_scores(fixed_f, mov_warped, mask_f, 
                                                      mask_m, do_hp, do_lp, do_bp, overlap_frac, tform)
            
        except Exception as e:
            warnings.warn(f"Transform {method} failed: {e}")
            traceback.print_exc()
        
        return pc_hp, pc_lp, pc_bp, tform, overlap_frac

    def _compute_hp_score(self, fixed_f, warped, mask_f, mask_m, tform, valid_mask):
        """Helper to compute HP Energy score for optimization."""
        from skimage import transform as sktransform
        cfg = self.config
        
        # Warp outlier mask
        mask_m_warped = sktransform.warp(
            mask_m.astype(np.float32),
            tform.inverse,
            output_shape=fixed_f.shape,
            order=0,
            preserve_range=True,
            mode='constant',
            cval=0.0,
        ) > 0.5
        
        final_valid = valid_mask & ~mask_f & ~mask_m_warped
        if final_valid.sum() < 200:
            return 0.0
            
        if final_valid.sum() < 200:
            return 0.0
            
        # Standard HP Correlation (Matches MATLAB corr2 logic)
        # To avoid boundary artifacts, perform the HP blurring on the VALID pixels mostly,
        # but since Gaussian filter is discrete, replacing NaNs with median avoids the sharp 0-edge artifact
        def _filter_safe(img, mask_val, sigma):
            mask = np.isnan(img)
            if not mask.any():
                return gaussian_filter(img, sigma)
            
            # Fill NaNs with the local or global median/mean instead of 0 to avoid huge artificial edges
            safe = img.copy()
            fill_val = np.nanmedian(img) if np.any(~mask) else 0
            safe[mask] = fill_val
            return gaussian_filter(safe, sigma)

        fix_hp = fixed_f - _filter_safe(fixed_f, mask_f, cfg.blur_hp)
        mov_hp = warped - _filter_safe(warped, mask_m_warped, cfg.blur_hp)
        
        # Extract valid pixels
        f_vec = fix_hp[final_valid]
        m_vec = mov_hp[final_valid]
        
        # Pearson Correlation
        if f_vec.std() == 0 or m_vec.std() == 0:
            return 0.0
            
        score = np.corrcoef(f_vec, m_vec)[0, 1]
        
        if np.isnan(score):
            return 0.0
            
        return score

    def _compute_scores(self, fixed_f, mov_warped, mask_f, mask_m, do_hp, do_lp, do_bp, 
                       overlap_frac, tform=None) -> Tuple[float, float, float]:
        """Compute all scores for a given alignment."""
        from skimage import transform as sktransform
        cfg = self.config
        
        # Base overlap mask
        valid_mask = np.isfinite(mov_warped) & np.isfinite(fixed_f)
        
        # Exclude outlier masks
        if tform is not None:
            mask_m_warped = sktransform.warp(
            mask_m.astype(np.float32),
            tform.inverse,
            output_shape=fixed_f.shape,
            order=0,
            preserve_range=True,
            mode='constant',
            cval=0.0,
        ) > 0.5
        else:
            mask_m_warped = mask_m
            
        final_valid = valid_mask & ~mask_f & ~mask_m_warped
        
        def zscore(x):
            return (x - x.mean()) / (x.std() + 1e-10)
            
        pc_hp, pc_lp, pc_bp = -np.inf, -np.inf, -np.inf
        
        if do_hp:
            pc_hp = self._compute_hp_score(fixed_f, mov_warped, mask_f, mask_m, 
                                          tform or sktransform.AffineTransform(), valid_mask)
        
        if do_lp and final_valid.sum() > 100:
            fix_lp = gaussian_filter(fixed_f, cfg.blur_lp)
            mov_lp = gaussian_filter(mov_warped, cfg.blur_lp)
            pc_lp = np.corrcoef(zscore(fix_lp[final_valid]), zscore(mov_lp[final_valid]))[0, 1]
            
        if do_bp and final_valid.sum() > 100:
            fix_bp = gaussian_filter(fixed_f, cfg.blur_bp1) - gaussian_filter(fixed_f, cfg.blur_bp2)
            mov_bp = gaussian_filter(mov_warped, cfg.blur_bp1) - gaussian_filter(mov_warped, cfg.blur_bp2)
            pc_bp = np.corrcoef(zscore(fix_bp[final_valid]), zscore(mov_bp[final_valid]))[0, 1]
            
        # Apply soft overlap penalty
        w = (overlap_frac - cfg.min_overlap_hard) / max(1e-6, 1 - cfg.min_overlap_hard)
        w = max(0, min(1, w)) ** cfg.gamma_overlap
        
        return pc_hp * w, pc_lp * w, pc_bp * w
    
    def _apply_transform(self, 
                        img: np.ndarray, 
                        tform: Any,
                        output_shape: Tuple[int, int]) -> np.ndarray:
        """Apply transform to image."""
        from skimage import transform as sktransform
        
        return sktransform.warp(
            img,
            tform.inverse,
            output_shape=output_shape,
            order=1,
            preserve_range=True,
            mode='constant',
            cval=np.nan,
        )


# ============================================================================ #
#                              CELLREGPY MAIN CLASS                            #
# ============================================================================ #

class CellRegPy:
    """
    Main cell registration pipeline.
    
    Ports batchRunCellReg.m to Python.
    
    Usage:
        config = CellRegConfig(microns_per_pixel=2.0)
        cellreg = CellRegPy(config)
        mouse_table, mouse_data = cellreg.run([Path('/path/to/mouse1')])
    """
    
    def __init__(self, config: Optional[CellRegConfig] = None):
        """
        Initialize CellRegPy.
        
        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or CellRegConfig()
        self.aligner = MeanImageAligner(self.config)
        
    def run(self, 
            mouse_folders: List[Union[str, Path]],
            rerun_skip_cellreg: bool = False,
            rerun_only_table: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the full cell registration pipeline.
        
        Mirrors MATLAB batchRunCellReg().
        
        Args:
            mouse_folders: List of paths to mouse data folders
            rerun_skip_cellreg: Skip alignment, regenerate mouse_data
            rerun_only_table: Only regenerate mouse_table
            
        Returns:
            Tuple of (mouse_table DataFrame, mouse_data dict)
        """
        mouse_folders = [Path(p) for p in mouse_folders]
        
        all_mouse_tables = []
        all_mouse_data = {}
        
        for mouse_folder in mouse_folders:
            print(f"Processing: {mouse_folder}")
            
            if not rerun_skip_cellreg and not rerun_only_table:
                # Full pipeline
                self._run_single_mouse(mouse_folder)
            
            # Build mouse_data and mouse_table
            sessions = self._get_sessions_from_folder(mouse_folder)
            mouse_name = mouse_folder.name
            
            if rerun_only_table:
                # Only rebuild the table; prefer loading existing mouse_data if available
                md_path = mouse_folder / 'mouse_data.npy'
                if md_path.exists():
                    try:
                        mouse_data = np.load(md_path, allow_pickle=True).item()
                    except Exception:
                        mouse_data = self._build_mouse_data(sessions, mouse_name, skip_cellreg=False)
                else:
                    mouse_data = self._build_mouse_data(sessions, mouse_name, skip_cellreg=False)
            else:
                # Build fresh mouse_data (always include CellReg results if present)
                mouse_data = self._build_mouse_data(sessions, mouse_name, skip_cellreg=False)

            mouse_table = self._build_cell_table(mouse_data)
            
            # Store
            all_mouse_data[mouse_name] = mouse_data
            all_mouse_tables.append(mouse_table)
            
            # Save
            self._save_results(mouse_folder, mouse_data, mouse_table)
        
        # Combine tables
        if all_mouse_tables:
            combined_table = pd.concat(all_mouse_tables, ignore_index=True)
        else:
            combined_table = pd.DataFrame()
        
        return combined_table, all_mouse_data
    
    def _run_single_mouse(self, mouse_folder: Path):
        """
        Run cell registration for a single mouse.
        
        This is the main loop from batchRunCellReg.m (lines 144-836).
        """
        print(f"Beginning processing: {mouse_folder}")
        
        # Get session folders
        plane0_folders = list_session_folders(mouse_folder)
        cellreg_files = get_cellreg_files(plane0_folders)
        
        if not cellreg_files:
            print(f"No CellReg.mat files found in {mouse_folder}")
            return

        # field of view, saved as CellReg.mat
        sess_fovs = cellreg_files
        if bool(getattr(self.config, 'test_run', False)):
            test_run_type = getattr(self.config, 'test_run_type', 'test random alignment')
            if test_run_type == 'test random alignment':
                # random alignment of 4 sessions
                print("Random alignment of 4 sessions")
                import random
                sess_fovs_temp = random.sample(sess_fovs, 4)
                sess_fovs = sess_fovs_temp
            elif test_run_type == 'test difficult alignment':
                # grab the first and last 3 sessions
                print("Difficult alignment of first and last 3 sessions")
                sess_fovs = [sess_fovs[0],[sess_fovs[-1:-3]]]
            else:
                print("Test run enabled: limiting to 4 sessions")
                sess_fovs = sess_fovs[:4]
        
        # Create results directory
        results_dir = mouse_folder / '1_CellReg'
        results_dir.mkdir(exist_ok=True)
        figures_dir = results_dir / 'Figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Step 1: Load spatial footprints
        spatial_footprints = self.step1_load_sessions(sess_fovs, figures_dir)
        
        # Load mean images
        print("Loading mean images and iscell data...")
        mean_images = []
        iscell_list = []
        for sess in sess_fovs:
            plane0_path = sess.parent
            mean_images.append(get_mean_image(plane0_path))
            iscell_list.append(get_iscell(plane0_path))
        
        # Optional spatial-footprint projection images (used for alignability + 2-stage refinement)
        footprint_images = None
        fallback_mode = str(getattr(self.config, 'alignment_fallback_mode', 'two_stage')).strip().lower()
        if fallback_mode in ('mean_then_footprints', 'footprints', 'two_stage'):
            print("Building spatial-footprint projection images (iscell-filtered)...")
            fp_arrays = []
            for sess in sess_fovs:
                fp_arrays.append(get_spatial_footprints(sess))
            fp_arrays, _, _, _, _ = adjust_fov_size(fp_arrays)

            footprint_images = []
            for fp, iscell in zip(fp_arrays, iscell_list):
                fp_use = fp
                try:
                    mask = np.asarray(iscell).astype(bool).squeeze()
                    if mask.ndim == 1 and fp.shape[0] == mask.shape[0]:
                        fp_use = fp[mask]
                except Exception:
                    pass
                footprint_images.append(
                    make_alignment_image_from_footprints(
                        fp_use,
                        pixel_weight_threshold=float(getattr(self.config, 'footprint_projection_threshold', 0.5)),
                    )
                )

        # Get alignable sessions
        alignable = self.get_alignable_sessions(mean_images, sess_fovs, footprint_images=footprint_images)
        
        # Plot alignable sessions graph
        if self.config.figures_visibility == 'on':
            plot_path = results_dir / 'alignable_sessions_graph.png'
            print(f"Saving alignment graph to {plot_path}")
            self.plot_alignable_sessions(alignable, save_path=plot_path)
        
        # Save alignable sessions
        np.save(results_dir / 'alignable_sessions.npy', alignable)
        
        # Remove redundancies if configured
        if self.config.remove_redundancies:
            alignable = self._remove_redundancies(alignable)
        
        # Process each FOV
        for fovi, fov_sessions in enumerate(alignable['session_names']):
            if len(fov_sessions) < 2:
                continue
                
            print(f"Working on FOV {fovi + 1}/{len(alignable['session_names'])}")
            
            # Create FOV directory
            fov_dir = results_dir / f'FOV{fovi + 1}'
            fov_dir.mkdir(exist_ok=True)
            fov_figures = fov_dir / 'Figures'
            fov_results = fov_dir / 'Results'
            fov_figures.mkdir(exist_ok=True)
            fov_results.mkdir(exist_ok=True)
            
            # Run registration for this FOV
            self._register_fov(
                fov_sessions,
                alignable['session_reference'][fovi],
                mean_images,
                alignable['index_aligned'][fovi],
                fov_results
            )
    
    def step1_load_sessions(self, 
                           sess_fovs: List[Path], 
                           figures_dir: Path) -> List[Path]:
        """
        Load spatial footprints for all sessions.
        
        Mirrors MATLAB step1.m.
        
        Args:
            sess_fovs: List of paths to CellReg.mat files
            figures_dir: Directory to save figures
            
        Returns:
            List of paths to spatial footprint files (memory efficient mode)
        """
        print(f"Stage 1: Loading {len(sess_fovs)} sessions")
        
        if self.config.memory_efficient_run:
            # Just return paths for lazy loading
            spatial_footprints = sess_fovs
        else:
            # Load all footprints into memory
            spatial_footprints = []
            for sess in sess_fovs:
                fp = get_spatial_footprints(sess)
                spatial_footprints.append(fp)
        
        # Count cells per session
        #cell_counts = []
        #for sess in sess_fovs:
        #    plane0_path = sess.parent
        #    iscell = get_iscell(plane0_path)
        #    cell_counts.append(iscell.sum())
        
        # Find session with most cells (for reference)
        #best_idx       = np.argmax(cell_counts)
        #seeded_session = sess_fovs[best_idx]
        
        #print(f"Session with most cells: {seeded_session.parent.name} ({cell_counts[best_idx]} cells)")
        
        return spatial_footprints
    
    def get_alignable_sessions(self,
                               mean_images: List[np.ndarray],
                               sess_names: List[Path],
                               footprint_images: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Find sessions that can be aligned based on mean image correlation.
        
        Mirrors MATLAB get_alignable_sessions.m.
        
        Args:
            mean_images: List of mean images
            sess_names: List of session paths/names
            
        Returns:
            Dictionary with alignment information per seed session
        """
        print("Finding alignable sessions...")
        
        n_sessions = len(sess_names)
        
        alignable = {
            'session_reference': [],
            'session_names': [],
            'correlations': [],
            'all_correlations': [], # Store full matrix for debugging
            'transformations': [],
            'alignment_sources': [],
            'index_aligned': [],
            'not_alignable': []
        }
        
        # Check each session as potential reference
        for align_i in range(n_sessions):
            print(f"  Testing session {align_i + 1}/{n_sessions} as reference")
            
            best_peaks = []
            best_tforms = []
            best_sources = []
            
            # Compare to all other sessions
            if self.config.use_parallel_processing:
                # Parallel processing
                try:
                    with mp.Pool() as pool:
                        args = [(mean_images, align_i, fov_i, footprint_images) 
                                for fov_i in range(n_sessions)]
                        results = pool.starmap(self._check_alignment, args)

                    best_sources = []
                    for _, _, peak, tform, source in results:
                        best_peaks.append(peak)
                        best_tforms.append(tform)
                        best_sources.append(source)
                except Exception as e:
                    print(f"    [WARN] multiprocessing failed ({e}); falling back to serial")
                    for fov_i in range(n_sessions):
                        print(f"    Comparing to session {fov_i + 1}/{n_sessions}")
                        _, _, peak, tform, source = self._check_alignment(mean_images, align_i, fov_i, footprint_images)
                        best_peaks.append(peak)
                        best_tforms.append(tform)
                        best_sources.append(source)

            else:
                # Serial processing
                for fov_i in range(n_sessions):
                    print(f"    Comparing to session {fov_i + 1}/{n_sessions}")
                    _, _, peak, tform, source = self._check_alignment(
                        mean_images, align_i, fov_i, footprint_images
                    )
                    best_peaks.append(peak)
                    best_tforms.append(tform)
                    best_sources.append(source)
            
            best_peaks = np.array(best_peaks)
            
            # Threshold
            thr = float(getattr(self.config, 'correlation_threshold', getattr(self.config, 'alignable_threshold', 0.0)))
            idx_align = np.where(best_peaks > thr)[0]
            
            alignable['session_reference'].append(sess_names[align_i])
            alignable['session_names'].append([sess_names[i] for i in idx_align])
            alignable['correlations'].append(best_peaks[idx_align])
            alignable['all_correlations'].append(best_peaks) # Store all
            alignable['transformations'].append([best_tforms[i] for i in idx_align])
            alignable['alignment_sources'].append([best_sources[i] for i in idx_align])
            alignable['index_aligned'].append(idx_align)
            alignable['not_alignable'].append(
                np.where(best_peaks <= thr)[0]
            )
        
        return alignable
    
    def plot_alignable_sessions(self, alignable: Dict, save_path: Optional[Path] = None):
        """
        Plot the alignable sessions graph and correlation matrix.
        
        Args:
            alignable: Dictionary returned by get_alignable_sessions
            save_path: Optional path to save the figure
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        n_sessions = len(alignable['session_reference'])
        if n_sessions == 0:
            print("No sessions to plot.")
            return

        # Prepare correlation matrix
        corr_mat = np.zeros((n_sessions, n_sessions))
        # Fill with alignable correlations
        for i, ref in enumerate(alignable['session_reference']):
            # Get indices of sessions aligned to this ref
            aligned_idxs = alignable['index_aligned'][i]
            corrs = alignable['correlations'][i]
            
            for target_idx, corr in zip(aligned_idxs, corrs):
                corr_mat[i, target_idx] = corr
        
        # Plot
        fig = plt.figure(figsize=(15, 6))
        
        # 1. Connectivity Graph
        ax1 = plt.subplot(1, 2, 1)
        G = nx.DiGraph()
        
        # Add nodes
        sess_labels = {i: Path(name).name[-8:] for i, name in enumerate(alignable['session_reference'])} # Short names
        G.add_nodes_from(range(n_sessions))
        
        # Add edges (only reliable alignments)
        for i in range(n_sessions):
            for j in range(n_sessions):
                if corr_mat[i, j] > self.config.alignable_threshold:
                    G.add_edge(i, j, weight=corr_mat[i, j])
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, labels=sess_labels, ax=ax1, font_size=8)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=weights, edge_cmap=plt.cm.viridis, width=1.5)
        
        ax1.set_title(f"Alignment Connectivity (Threshold > {self.config.alignable_threshold})")
        ax1.axis('off')
        
        # 2. Correlation Matrix
        ax2 = plt.subplot(1, 2, 2)
        im = ax2.imshow(corr_mat, cmap='viridis', vmin=0, vmax=1)
        ax2.set_title("Pairwise Correlation Matrix")
        ax2.set_xlabel("Target Session")
        ax2.set_ylabel("Reference Session")
        plt.colorbar(im, ax=ax2, label="Correlation")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def check_alignment_quality(self, alignable: Dict, mean_images: List[np.ndarray]):
        """
        Visual check of alignment quality.
        Plots aligned vs non-aligned examples for each session.
        
        Args:
            alignable: Dictionary from get_alignable_sessions
            mean_images: List of mean images
        """
        import matplotlib.pyplot as plt
        
        print("\n=== Alignment Quality Check ===")
        
        for i, ref_name in enumerate(alignable['session_reference']):
             # Short name for display
            short_ref = Path(ref_name).name[-15:]
            ref_img = mean_images[i]
            
            good_idxs = alignable['index_aligned'][i]
            bad_idxs = alignable['not_alignable'][i]
            
            # Get correlations if available (new field)
            all_corrs = alignable.get('all_correlations', [])
            if len(all_corrs) > i:
                corrs_i = all_corrs[i]
            else:
                corrs_i = None

            print(f"\nSession {i+1} ({short_ref}):")
            
            if corrs_i is not None:
                # Format with scores - Use 1-based indexing for display to match "Sess X"
                good_str = ", ".join([f"Sess{idx+1}({corrs_i[idx]:.2f})" for idx in good_idxs])
                bad_str = ", ".join([f"Sess{idx+1}({corrs_i[idx]:.2f})" for idx in bad_idxs])
                print(f"  Alignable ({len(good_idxs)}): [{good_str}]")
                print(f"  Not alignable ({len(bad_idxs)}): [{bad_str}]")
            else:
                good_str = ", ".join([f"Sess{idx+1}" for idx in good_idxs])
                bad_str = ", ".join([f"Sess{idx+1}" for idx in bad_idxs])
                print(f"  Alignable ({len(good_idxs)}): [{good_str}]")
                print(f"  Not alignable ({len(bad_idxs)}): [{bad_str}]")
            
            # Plot one Alignable example (if any exists other than self)
            others_good = [x for x in good_idxs if x != i]
            if others_good:
                target_idx = others_good[0]
                print(f"  -> Plotting Alignable Example: vs Session {target_idx+1}")
                self.aligner.plot_image_pair(
                    ref_img, 
                    mean_images[target_idx], 
                    title=f"ALIGNABLE: {short_ref} vs Sess {target_idx+1}"
                )
            
            # Plot one Not Alignable example
            if len(bad_idxs) > 0:
                target_idx = bad_idxs[0]
                print(f"  -> Plotting Non-Alignable Example: vs Session {target_idx+1}")
                self.aligner.plot_image_pair(
                    ref_img, 
                    mean_images[target_idx], 
                    title=f"NOT ALIGNABLE: {short_ref} vs Sess {target_idx+1}"
                )
    
    def _alignment_needs_fallback(self, peak: float) -> bool:
        """Return True when the primary alignment score is not usable."""
        try:
            peak_f = float(peak)
        except Exception:
            return True
        if not np.isfinite(peak_f):
            return True
        return peak_f < float(self.config.alignable_threshold)

    def _align_with_optional_fallback(self,
                                          fixed_img: np.ndarray,
                                          moving_img: np.ndarray,
                                          *,
                                          fixed_fp_img: Optional[np.ndarray] = None,
                                          moving_fp_img: Optional[np.ndarray] = None,
                                          plot_fig: bool = False):
        """Mean-image alignment with optional footprint alignment.

        Modes (cfg.alignment_fallback_mode):
            - 'two_stage' (default): mean-image alignment first, then refine using
              spatial-footprint projection alignment (iscell-filtered).
            - 'mean_then_footprints': only try footprint alignment if mean-image
              alignment is below threshold.
            - 'footprints': footprint alignment only.
            - 'none': mean-image alignment only.

        Returns:
            registered, method, peak, tform, filter_name, outliers, source
        """
        cfg = self.config
        mode = str(getattr(cfg, 'alignment_fallback_mode', 'two_stage')).strip().lower()

        from skimage import transform as sktransform

        # ------------------------- Stage 1: mean images -------------------------
        reg1, method1, peak1, tform1, filt1, out1 = self.aligner.align(
            fixed_img,
            moving_img,
            filter_mode=getattr(cfg, 'filter_mode', 'highpass'),
            outlier_mode=getattr(cfg, 'outlier_mode', 'off'),
            plot_fig=plot_fig,
        )

        # Early exits
        if mode in ('none', 'mean'):
            return reg1, method1, peak1, tform1, filt1, out1, 'mean_image'
        if mode == 'mean_then_footprints' and (not self._alignment_needs_fallback(peak1)):
            return reg1, method1, peak1, tform1, filt1, out1, 'mean_image'

        # ------------------------- Footprint-only mode -------------------------
        if mode == 'footprints':
            if fixed_fp_img is None or moving_fp_img is None:
                raise ValueError("cfg.alignment_fallback_mode='footprints' but footprint images were not provided.")
            reg2, method2, peak2, tform2, filt2, out2 = self.aligner.align(
                fixed_fp_img,
                moving_fp_img,
                filter_mode=getattr(cfg, 'footprint_filter_mode', 'highpass'),
                outlier_mode=getattr(cfg, 'footprint_outlier_mode', 'off'),
                plot_fig=plot_fig,
            )
            return reg2, f"fp:{method2}", peak2, tform2, filt2, out2, 'spatial_footprints'

        # ------------------------- Stage 2: footprints -------------------------
        if fixed_fp_img is None or moving_fp_img is None:
            # Can't do stage 2
            return reg1, method1, peak1, tform1, filt1, out1, 'mean_image'

        # Pre-warp moving footprints with stage-1 transform for refinement
        moving_fp_for_stage2 = moving_fp_img
        tform1_ok = (tform1 is not None)
        if tform1_ok:
            try:
                moving_fp_for_stage2 = sktransform.warp(
                    np.asarray(moving_fp_img, dtype=float),
                    tform1.inverse,
                    output_shape=np.asarray(fixed_fp_img).shape,
                    order=1,
                    preserve_range=True,
                    mode='constant',
                    cval=0.0,
                )
            except Exception:
                # If pre-warp fails, just align raw footprint projections
                moving_fp_for_stage2 = moving_fp_img
                tform1_ok = False

        reg2, method2, peak2, tform2, filt2, out2 = self.aligner.align(
            fixed_fp_img,
            moving_fp_for_stage2,
            filter_mode=getattr(cfg, 'footprint_filter_mode', 'highpass'),
            outlier_mode=getattr(cfg, 'footprint_outlier_mode', 'off'),
            plot_fig=plot_fig,
        )

        # Compose transforms: T = T2 ∘ T1
        t_final = tform1
        if tform2 is not None:
            try:
                if tform1_ok:
                    mat = np.asarray(tform2.params) @ np.asarray(tform1.params)
                else:
                    mat = np.asarray(tform2.params)
                t_final = sktransform.AffineTransform(matrix=mat)
            except Exception:
                # Fallback: use stage-2 transform if composition fails
                t_final = tform2

        # Decide whether to accept footprint refinement.
        # Default: accept if the footprint-stage score is finite and >= the mean-image score,
        # or if the mean-image score is unusable (NaN / -inf).
        try:
            peak1_f = float(peak1)
        except Exception:
            peak1_f = float('nan')
        try:
            peak2_f = float(peak2)
        except Exception:
            peak2_f = float('nan')

        peak2_ok = np.isfinite(peak2_f) and (peak2_f != -np.inf)
        peak1_ok = np.isfinite(peak1_f) and (peak1_f != -np.inf)

        accept_refine = peak2_ok and ((not peak1_ok) or (peak2_f >= peak1_f))

        if accept_refine:
            # For gating/alignability, report the *best* score we observed.
            peak_final = peak2_f
            if peak1_ok:
                peak_final = max(peak_final, peak1_f)
            method = f"mean:{method1}->fp:{method2}"
            filt = f"{filt1}->{filt2}"
            outliers = (out1, out2)
            source = 'mean_then_footprints'
            reg = reg2
        else:
            peak_final = peak1_f
            t_final = tform1
            method = f"mean:{method1} (fp_refine_rejected:{method2})"
            filt = filt1
            outliers = out1
            source = 'mean_image'
            reg = reg1

        # Preserve original behavior when peaks are not finite
        peak_return = peak_final
        if not np.isfinite(peak_return) or peak_return == -np.inf:
            peak_return = peak1

        return reg, method, peak_return, t_final, filt, outliers, source

    def _check_alignment(self, 
                        mean_images: List[np.ndarray],
                        ref_idx: int,
                        mov_idx: int,
                        footprint_images: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, str, float, Any, str]:
        """Check alignment between two sessions, with optional footprint-projection fallback."""
        fixed_fp = None
        moving_fp = None
        if footprint_images is not None:
            fixed_fp = footprint_images[ref_idx]
            moving_fp = footprint_images[mov_idx]

        _, method, peak, tform, _, _, source = self._align_with_optional_fallback(
            mean_images[ref_idx],
            mean_images[mov_idx],
            fixed_fp_img=fixed_fp,
            moving_fp_img=moving_fp,
            plot_fig=False,
        )
        return None, method, peak, tform, source
    
    def _remove_redundancies(self, alignable: Dict) -> Dict:
        """
        Remove redundant FOV alignments.
        
        Mirrors MATLAB redundancy removal logic (lines 217-286).
        """
        print("Removing redundancies...")
        
        # Get median correlations
        median_corrs = [np.median(c) if len(c) > 0 else -np.inf 
                       for c in alignable['correlations']]
        median_corrs = np.array(median_corrs)
        median_corrs[np.isnan(median_corrs)] = -np.inf
        
        # Normalize index sets
        index_sets = [set(np.sort(np.unique(idx)).tolist()) 
                     for idx in alignable['index_aligned']]
        
        # Convert to frozensets for comparison
        frozen_sets = [frozenset(s) for s in index_sets]
        
        # Find identical sets - keep best correlation
        unique_sets = {}
        for i, fs in enumerate(frozen_sets):
            key = fs
            if key in unique_sets:
                if median_corrs[i] > median_corrs[unique_sets[key]]:
                    unique_sets[key] = i
            else:
                unique_sets[key] = i
        
        keep_indices = list(unique_sets.values())
        
        # Remove strict subsets
        to_remove = set()
        for i in keep_indices:
            for j in keep_indices:
                if i == j:
                    continue
                if frozen_sets[i] < frozen_sets[j]:  # strict subset
                    to_remove.add(i)
                    break
        
        final_keep = [i for i in keep_indices if i not in to_remove]
        
        # Filter alignable dict
        filtered = {}
        for key in alignable:
            filtered[key] = [alignable[key][i] for i in final_keep]
        
        print(f"  Kept {len(final_keep)}/{len(alignable['session_names'])} FOVs")
        
        return filtered
    
    def _register_fov(self,
                     sessions: List[Path],
                     reference_session: Path,
                     mean_images: List[np.ndarray],
                     session_indices: np.ndarray,
                     results_dir: Path):
        """
        Register cells for one FOV.
        
        This is the main registration logic from batchRunCellReg.m (lines 374-836).
        
        Args:
            sessions: List of session paths in this FOV
            reference_session: Reference session path
            mean_images: List of mean images (full list, use session_indices)
            session_indices: Indices into mean_images for this FOV
            results_dir: Directory to save results
        """
        cfg = self.config

        # Dual-model settings (Stage 6 style): centroid-primary + spatial-correlation floor veto
        use_dual_model = bool(getattr(cfg, 'dual_model', False) or getattr(cfg, 'apply_spatial_floor_filter', False))
        spatial_corr_floor = float(getattr(cfg, 'spatial_corr_floor', 0.5))
        n_sessions = len(sessions)
        print(f"  Registering {n_sessions} sessions...")
        
        # Load spatial footprints for this FOV
        spatial_footprints = []
        iscell_list = []
        for sess_path in sessions:
            fp = get_spatial_footprints(sess_path)
            spatial_footprints.append(fp)
            
            plane0_path = sess_path.parent
            iscell_list.append(get_iscell(plane0_path))
        
        # Select mean images for this FOV
        fov_mean_images = [mean_images[i] for i in session_indices]
        
        # Step 2a: Normalize spatial footprints
        temp_dir = results_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        if cfg.memory_efficient_run:
            normalized_fps = normalize_footprints(spatial_footprints, write_path=temp_dir)
            # Reload as arrays
            normalized_fps = [np.load(p) for p in normalized_fps]
        else:
            normalized_fps = normalize_footprints(spatial_footprints)
        
        # Step 2b: Adjust FOV sizes
        adjusted_fps, adjusted_fov, adj_x, adj_y, padding = adjust_fov_size(normalized_fps)
        del normalized_fps  # Free memory
        
        # Step 2c: Compute projections and centroids
        footprint_projections = compute_footprint_projections(adjusted_fps)
        footprint_alignment_images = []
        for fp, iscell in zip(adjusted_fps, iscell_list):
            fp_use = fp
            try:
                mask = np.asarray(iscell).astype(bool).squeeze()
                if mask.ndim == 1 and fp.shape[0] == mask.shape[0]:
                    fp_use = fp[mask]
            except Exception:
                pass
            footprint_alignment_images.append(
                make_alignment_image_from_footprints(
                    fp_use,
                    pixel_weight_threshold=float(getattr(cfg, 'footprint_projection_threshold', 0.5)),
                )
            )
        centroid_locations = compute_centroids(adjusted_fps, cfg.microns_per_pixel)
        centroid_projections = compute_centroid_projections(centroid_locations, adjusted_fps)
        
        # Step 2d: Align images to reference
        # Find reference session index in this FOV
        ref_idx = 0  # Default to first session
        for i, sess in enumerate(sessions):
            if sess == reference_session:
                ref_idx = i
                break

        # Align footprints using mean-image transformations (matches demo_validate_alignment)
        print("  Aligning cells to reference coordinate system...")
        aligned_fps = [None] * n_sessions
        aligned_centroid_locations = [None] * n_sessions

        # Book-keeping for validation plots
        alignment_translations = np.zeros((3, n_sessions), dtype=float)  # [dx; dy; rot_deg]
        maximal_cross_correlation = np.full(n_sessions, np.nan, dtype=float)
        alignment_sources = np.array(['reference'] * n_sessions, dtype=object)
        
        # Mean-image correlation diagnostics (raw vs aligned)
        def _safe_flat_corr(a, b) -> float:
            try:
                aa = np.asarray(a, dtype=float).ravel()
                bb = np.asarray(b, dtype=float).ravel()
                if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
                    return float('nan')
                aa = aa - np.nanmean(aa)
                bb = bb - np.nanmean(bb)
                denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
                if denom == 0.0 or (not np.isfinite(denom)):
                    return float('nan')
                return float(np.dot(aa, bb) / denom)
            except Exception:
                return float('nan')
        
        raw_image_correlation = np.full(n_sessions, np.nan, dtype=float)
        aligned_image_correlation = np.full(n_sessions, np.nan, dtype=float)
        raw_image_correlation[ref_idx] = 1.0
        aligned_image_correlation[ref_idx] = 1.0

        from skimage import transform as sktransform

        for i in range(n_sessions):
            if i == ref_idx:
                aligned_fps[i] = adjusted_fps[i]
                aligned_centroid_locations[i] = centroid_locations[i]
                maximal_cross_correlation[i] = 1.0
                continue

            # Correlation before alignment (mean images)
            raw_image_correlation[i] = _safe_flat_corr(fov_mean_images[ref_idx], fov_mean_images[i])

            # Get transformation from mean image alignment
            _, method, peak, tform, _, _, alignment_source = self._align_with_optional_fallback(
                fov_mean_images[ref_idx],
                fov_mean_images[i],
                fixed_fp_img=footprint_alignment_images[ref_idx],
                moving_fp_img=footprint_alignment_images[i],
            )
            alignment_sources[i] = alignment_source

            # Normalize peak into a safe float for downstream plots
            try:
                peak_f = float(peak)
            except Exception:
                peak_f = float("nan")
            if not np.isfinite(peak_f) or peak_f == -np.inf:
                peak_f = float("nan")
            maximal_cross_correlation[i] = peak_f

            # Correlation after applying the selected transform to the mean image
            if tform is not None:
                try:
                    reg_mean = sktransform.warp(
                        np.asarray(fov_mean_images[i], dtype=float),
                        tform.inverse,
                        output_shape=np.asarray(fov_mean_images[ref_idx]).shape,
                        order=1, preserve_range=True, mode='constant', cval=0.0,
                    )
                    aligned_image_correlation[i] = _safe_flat_corr(fov_mean_images[ref_idx], reg_mean)
                except Exception:
                    aligned_image_correlation[i] = raw_image_correlation[i]
            else:
                aligned_image_correlation[i] = raw_image_correlation[i]

            if tform is not None and np.isfinite(peak_f) and peak_f >= float(cfg.alignable_threshold):
                # Extract rigid-ish params for reporting (dx, dy, rot)
                try:
                    params = np.asarray(tform.params, dtype=float)
                    alignment_translations[0, i] = float(params[0, 2])
                    alignment_translations[1, i] = float(params[1, 2])
                    angle_rad = float(np.arctan2(params[1, 0], params[0, 0]))
                    alignment_translations[2, i] = float(np.degrees(angle_rad))
                except Exception:
                    pass

                # Apply transformation to footprints
                n_cells = adjusted_fps[i].shape[0]
                aligned = np.zeros_like(adjusted_fps[i])
                for c in range(n_cells):
                    aligned[c] = sktransform.warp(
                        adjusted_fps[i][c], tform.inverse,
                        output_shape=adjusted_fps[i][c].shape,
                        order=1, preserve_range=True, mode='constant', cval=0.0,
                    )
                aligned_fps[i] = aligned

                # Transform centroids (centroids are (x, y); skimage expects (x, y))
                cents = centroid_locations[i]
                if len(cents) > 0:
                    coords = np.column_stack([cents[:, 0], cents[:, 1], np.ones(len(cents))])
                    transformed = (np.asarray(tform.params) @ coords.T).T
                    aligned_centroid_locations[i] = transformed[:, :2]
                else:
                    aligned_centroid_locations[i] = cents

                print(
                    f"    Session {i+1}: source={alignment_source}, method={method}, peak={peak_f:.3f}, "
                    f"dx={alignment_translations[0,i]:.1f}, dy={alignment_translations[1,i]:.1f}, "
                    f"rot={alignment_translations[2,i]:.2f}°"
                )
            else:
                # Below threshold (or no transform) – keep as-is
                aligned_fps[i] = adjusted_fps[i]
                aligned_centroid_locations[i] = centroid_locations[i]
                print(f"    Session {i+1}: source={alignment_source}, method={method}, peak={peak_f:.3f} (below threshold)")

        # ------------------------------------------------------------------
        # Peak-gated SIMPLE registration (IoU + Hungarian).
        #
        # When the mean-image alignment peak score is extremely high, CellReg's
        # probabilistic mixture modeling (and especially the neighbor displacement
        # cloud) can become misleading because there are many "wrong but nearby"
        # ROI pairs in dense FOVs. In this regime we skip CellReg modeling
        # entirely and do deterministic IoU+Hungarian registration on iscell ROIs.
        #
        # NOTE: This block intentionally does NOT sweep / change cfg.maximal_distance.
        # ------------------------------------------------------------------
        use_flex = False
        skip_probabilistic = False
        flex_debug = {}
        simple_candidate = None
        max_dist_px = None
        data_dist = None

        # Peak score computed by MeanImageAligner (maximal_cross_correlation per session)
        peak_mask = np.ones(n_sessions, dtype=bool)
        peak_mask[ref_idx] = False
        peak_med = np.nanmedian(maximal_cross_correlation[peak_mask]) if peak_mask.any() else float('nan')
        peak_thr = float(getattr(cfg, 'auto_flex_peak_threshold', 0.95))

        if bool(getattr(cfg, 'auto_flex_on_high_peak', False)) and np.isfinite(peak_med) and (peak_med >= peak_thr):
            use_flex = True
            skip_probabilistic = True
            print(f"  🧠 High alignment peak detected (median peak={peak_med:.3f} ≥ {peak_thr:.3f}).")
            print("  Switching to SIMPLE IoU+Hungarian registration (peak-gated; skipping CellReg modeling).")

            # Build iscell-filtered arrays (operate only on true cells)
            idx_maps = []
            fps_cells = []
            cents_cells = []
            for fp, cents, iscell in zip(aligned_fps, aligned_centroid_locations, iscell_list):
                mask = np.asarray(iscell).astype(bool).squeeze()
                if (mask.ndim == 1) and (fp.shape[0] == mask.shape[0]):
                    idx = np.where(mask)[0]
                    idx_maps.append(idx)
                    fps_cells.append(fp[mask])
                    cents_cells.append(cents[mask])
                else:
                    idx_maps.append(np.arange(fp.shape[0]))
                    fps_cells.append(fp)
                    cents_cells.append(cents)

            # IMPORTANT: do NOT change the user's maximal_distance; convert µm->px here.
            max_dist_px_simple = float(cfg.maximal_distance) / float(cfg.microns_per_pixel)

            cmap_c, cdist_c, iou_c, reg_d_c, nonreg_d_c = initial_registration_iou_hungarian(
                fps_cells,
                cents_cells,
                reference_session_index=ref_idx,
                maximal_distance=max_dist_px_simple,
                mask_threshold=float(getattr(cfg, 'simple_mask_threshold', 0.15)),
                iou_threshold=float(getattr(cfg, 'simple_iou_threshold', 0.10)),
                cost_beta=float(getattr(cfg, 'simple_cost_beta', 0.25)),
            )

            # Remap cell-only indices back to original ROI indices (suite2p indexing space, 1-indexed)
            cmap_full = np.asarray(cmap_c, dtype=int).copy()
            for s, idx in enumerate(idx_maps):
                nz = cmap_full[:, s] > 0
                if np.any(nz):
                    cmap_full[nz, s] = (idx[cmap_full[nz, s] - 1] + 1).astype(int)

            # Set outputs as FINAL
            cell_to_index_map = cmap_full
            centroid_distance_map = np.asarray(cdist_c, dtype=float)
            spatial_correlation_map = np.asarray(iou_c, dtype=float)  # reuse slot for IoU map
            registered_distances = np.asarray(reg_d_c, dtype=float)
            non_registered_distances = np.asarray(nonreg_d_c, dtype=float)
            registered_correlations = np.array([], dtype=float)
            non_registered_correlations = np.array([], dtype=float)

            # Minimal placeholders so the rest of the pipeline (saving + tables + figures) remains stable
            model_used = "Simple IoU+Hungarian (peak-gated)"
            initial_metric_threshold = float(getattr(cfg, 'simple_iou_threshold', 0.10))
            best_model_string = model_used
            # Ensure downstream saving never crashes (probabilistic fields are absent in simple mode)
            p_same_models = dict(
                simple_mode=True,
                method="iou_hungarian",
                peak_median=float(peak_med) if np.isfinite(peak_med) else float('nan'),
                peak_threshold=float(peak_thr),
                maximal_distance_um=float(getattr(cfg, 'maximal_distance', float('nan'))),
                microns_per_pixel=float(getattr(cfg, 'microns_per_pixel', float('nan'))),
                mask_threshold=float(getattr(cfg, 'simple_mask_threshold', 0.15)),
                iou_threshold=float(getattr(cfg, 'simple_iou_threshold', 0.10)),
                cost_beta=float(getattr(cfg, 'simple_cost_beta', 0.25)),
            )
            number_of_bins = 50
            centers_of_bins = (
                np.linspace(0, max_dist_px_simple, number_of_bins, dtype=np.float64),
                np.linspace(0, 1, number_of_bins, dtype=np.float64),
            )
            centroid_overlap_mse = float("nan")
            corr_overlap_mse = float("nan")
            centroid_intersection = float("nan")
            corr_intersection = float("nan")
            centroid_best_model = ""
            corr_best_model = ""
            centroid_same_model = np.array([])
            centroid_diff_model = np.array([])
            centroid_mixture_model = np.array([])
            corr_same_model = np.array([])
            corr_diff_model = np.array([])
            corr_mixture_model = np.array([])
            p_same_given_centroid_distance = np.array([])
            p_same_given_spatial_correlation = np.array([])
            p_same_centroid_distances = []
            p_same_spatial_correlations = []
            all_to_all_p_same = []
            registered_cells_centroids = None
            cluster_scores = {}
            pre_spatial_floor_map = cell_to_index_map.copy()
            n_vetoed_clusters = 0
            vetoed_centroid_um = []
            vetoed_spatial_corr = []
            p_same_vec = np.array([])
            p_different_vec = np.array([])
            accuracy_scores = np.array([np.nan, np.nan, np.nan], dtype=float)

            # Provide a simple_candidate dict so Stage 7 QC plots are produced (even though it's FINAL here)
            try:
                dx_f, dy_f = compute_registered_pair_displacements(cell_to_index_map, aligned_centroid_locations, ref_idx)
                q_f = displacement_quality(dx_f, dy_f, cfg.microns_per_pixel, target_um=2.0)
            except Exception:
                q_f = {}

            simple_candidate = dict(
                cell_to_index_map=cell_to_index_map,
                quality=q_f,
            )

            flex_debug = dict(
                peak_med=float(peak_med),
                peak_thr=float(peak_thr),
                method="iou_hungarian",
                max_dist_um=float(cfg.maximal_distance),
                max_dist_px=float(max_dist_px_simple),
                quality=q_f,
            )

        # (Continue with the standard probabilistic CellReg workflow below.)
        # (Continue with the standard probabilistic CellReg workflow below.)

        if not skip_probabilistic:

            # Step 3: Compute data distribution

            if max_dist_px is None:
                max_dist_px = cfg.maximal_distance / cfg.microns_per_pixel

            if data_dist is None:
                print("  Computing cell-pair similarity distributions...")
                data_dist = compute_data_distribution(
                    aligned_fps,
                    aligned_centroid_locations,
                    max_dist_px
                )
            else:
                print(f"  Using precomputed similarity distributions (max_dist_px={float(max_dist_px):.3f}).")

        
            # Step 4b: Probabilistic modeling and clustering (MATLAB CellReg-style)
            p_same_models = {}
            if cfg.registration_approach.strip().lower().startswith("prob"):
                # Bin centers (distance in pixels, correlation in [0,1])
                number_of_bins, _ = estimate_num_bins(aligned_fps, max_dist_px)
                centers_of_bins = (
                    np.linspace(0, max_dist_px, number_of_bins, dtype=np.float64),
                    np.linspace(0, 1, number_of_bins, dtype=np.float64),
                )

                # Fit mixture models from neighbor distributions
                (p_same_given_centroid_distance,
                 centroid_same_model,
                 centroid_diff_model,
                 centroid_mixture_model,
                 centroid_intersection,
                 centroid_best_model,
                 centroid_overlap_mse) = compute_centroid_distances_model_custom(
                    data_dist["neighbors_centroid_distances"], number_of_bins, centers_of_bins, 
                    microns_per_pixel=cfg.microns_per_pixel
                )

                (p_same_given_spatial_correlation,
                 corr_same_model,
                 corr_diff_model,
                 corr_mixture_model,
                 corr_intersection,
                 corr_best_model,
                 corr_overlap_mse) = compute_spatial_correlations_model(
                    data_dist["neighbors_spatial_correlations"], number_of_bins, centers_of_bins
                )

                # Convert all-to-all values into all-to-all p_same lookups
                p_same_centroid_distances, p_same_spatial_correlations = compute_p_same(
                    data_dist["all_to_all_centroid_distances"],
                    data_dist["all_to_all_spatial_correlations"],
                    centers_of_bins,
                    p_same_given_centroid_distance,
                    p_same_given_spatial_correlation,
                )

                # Choose which probabilistic model to use for clustering
            
                raw_model_type = str(cfg.model_type).strip()

                # Always compute best_model_string for reporting, but optionally override
                # the *final* registration to use the dual-model approach (Stage 6).

                if raw_model_type.lower() in ('auto', 'best', 'matlab'):
                    best_model_string = choose_best_model(
                        centroid_overlap_mse,
                        corr_overlap_mse,
                        centroid_intersection=centroid_intersection,
                        corr_intersection=corr_intersection,
                        prefer='Spatial correlation',
                    )
                else:
                    if raw_model_type.lower().startswith('centroid'):
                        best_model_string = 'Centroid distance'
                    elif raw_model_type.lower().startswith('spatial') or raw_model_type.lower().startswith('corr'):
                        best_model_string = 'Spatial correlation'
                    else:
                        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Use 'auto', 'Spatial correlation', or 'Centroid distance'.")

                # Dual-model override: force centroid-primary clustering, then apply spatial floor veto
                model_used = 'Centroid distance' if use_dual_model else best_model_string

                if model_used == 'Centroid distance':
                    all_to_all_p_same = p_same_centroid_distances
                    # NOTE: centroid_intersection is in µm; initial registration expects pixels
                    initial_metric_threshold = (float(centroid_intersection) / float(cfg.microns_per_pixel)) if np.isfinite(centroid_intersection) else max_dist_px
                else:
                    all_to_all_p_same = p_same_spatial_correlations
                    initial_metric_threshold = corr_intersection if np.isfinite(corr_intersection) else cfg.sufficient_correlation_footprints

            # Step 4: Initial registration (MATLAB: initial_registration_type = best_model_string)
            if model_used == "Centroid distance":
                (cell_to_index_map,
                 registered_distances,
                 non_registered_distances,
                 centroid_distance_map) = initial_registration_centroid_distances_custom(
                    aligned_centroid_locations,
                    maximal_distance=max_dist_px,
                    centroid_distance_threshold=initial_metric_threshold,
                )
                registered_correlations = np.array([])
                non_registered_correlations = np.array([])
                spatial_correlation_map = np.zeros_like(centroid_distance_map)
            else:
                (cell_to_index_map,
                 registered_correlations,
                 non_registered_correlations,
                 spatial_correlation_map) = initial_registration_spatial_corr(
                    aligned_fps,
                    aligned_centroid_locations,
                    maximal_distance=max_dist_px,
                    spatial_correlation_threshold=initial_metric_threshold,
                )
                centroid_distance_map = np.zeros_like(spatial_correlation_map)
                registered_distances = np.array([])
                non_registered_distances = np.array([])

            # Step 5: Cluster / refine mapping (MATLAB Stage 5)
            # We always cluster using the selected model_used. For the dual-model approach,
            # model_used is forced to 'Centroid distance' (centroid-primary).
            centroid_primary_map, registered_cells_centroids, cluster_scores = cluster_cells_matlab(
                cell_to_index_map,
                all_to_all_p_same,
                data_dist["all_to_all_indexes"],
                max_dist_px,
                cfg.p_same_threshold,
                aligned_centroid_locations,
                registration_approach="Probabilistic",
                transform_data=False,
                verbose=True
            )

            # Step 6 (dual-model): spatial floor veto (centroid-primary + spatial-corr cutoff)
            pre_spatial_floor_map = centroid_primary_map.copy()
            n_vetoed_clusters = 0
            vetoed_centroid_um = []
            vetoed_spatial_corr = []

            if use_dual_model:
                filtered_map = pre_spatial_floor_map.copy()
                for cluster_idx in range(filtered_map.shape[0]):
                    row = filtered_map[cluster_idx, :]
                    present_sessions = np.where(row > 0)[0]
                    if present_sessions.size < 2:
                        continue
                    veto = False
                    # check all within-cluster session pairs
                    for ii in range(present_sessions.size):
                        if veto: break
                        si = int(present_sessions[ii])
                        ci = int(row[si]) - 1
                        for jj in range(ii + 1, present_sessions.size):
                            sj = int(present_sessions[jj])
                            cj = int(row[sj]) - 1
                            fp_i = aligned_fps[si][ci]
                            fp_j = aligned_fps[sj][cj]
                            sc = compute_spatial_correlation(fp_i, fp_j)
                            if sc < spatial_corr_floor:
                                # record the failing pair for histograms
                                di = aligned_centroid_locations[si][ci]
                                dj = aligned_centroid_locations[sj][cj]
                                dpx = float(np.sqrt(np.sum((di - dj) ** 2)))
                                vetoed_centroid_um.append(dpx * float(cfg.microns_per_pixel))
                                vetoed_spatial_corr.append(float(sc))
                                veto = True
                                break
                    if veto:
                        # dissolve this multi-session cluster into singletons (keep first session only)
                        for s_idx in present_sessions[1:]:
                            filtered_map[cluster_idx, int(s_idx)] = 0
                        n_vetoed_clusters += 1

                cell_to_index_map = filtered_map
            else:
                cell_to_index_map = centroid_primary_map

            # Recompute probabilistic accuracy on FINAL map (post-veto if dual_model enabled)
            p_same_vec, p_different_vec, accuracy_scores = estimate_registration_accuracy(
                cell_to_index_map,
                all_to_all_p_same,
                data_dist["all_to_all_indexes"],
                threshold=cfg.p_same_threshold,
            )

            # Recompute cluster score distributions on FINAL map (so any downstream plots match the dual output)
            try:
                scores_out = compute_scores_matlab(
                    cell_to_index_map, data_dist["all_to_all_indexes"], all_to_all_p_same, n_sessions
                )
                cluster_scores = dict(
                    cell_scores=scores_out[0],
                    cell_scores_positive=scores_out[1],
                    cell_scores_negative=scores_out[2],
                    cell_scores_exclusive=scores_out[3],
                    p_same_registered_pairs=scores_out[4],
                )
            except Exception:
                pass

            p_same_models = dict(
                model_used=model_used,
                best_model_string=best_model_string if 'best_model_string' in locals() else model_used,
                dual_model=use_dual_model,
                spatial_corr_floor=spatial_corr_floor,
                n_vetoed_clusters=int(n_vetoed_clusters) if 'n_vetoed_clusters' in locals() else 0,
                number_of_bins=number_of_bins,
                centers_of_bins=centers_of_bins,
                p_same_threshold=cfg.p_same_threshold,
                # Centroid-distance model
                p_same_given_centroid_distance=p_same_given_centroid_distance,
                centroid_same_model=centroid_same_model,
                centroid_different_model=centroid_diff_model,
                centroid_mixture_model=centroid_mixture_model,
                centroid_intersection=centroid_intersection,
                centroid_best_model=centroid_best_model,
                centroid_overlap_mse=centroid_overlap_mse,
                # Spatial-correlation model
                p_same_given_spatial_correlation=p_same_given_spatial_correlation,
                spatial_same_model=corr_same_model,
                spatial_different_model=corr_diff_model,
                spatial_mixture_model=corr_mixture_model,
                spatial_intersection=corr_intersection,
                spatial_best_model=corr_best_model,
                spatial_overlap_mse=corr_overlap_mse,
                # Clustering outputs
                registered_cells_centroids=registered_cells_centroids,
                cluster_scores=cluster_scores,
                p_same_vec=p_same_vec,
                p_different_vec=p_different_vec,
                accuracy_scores=accuracy_scores,
            )

            # ------------------------------------------------------------------
            # Auto-flex: optionally choose between the probabilistic result and the IoU+Hungarian candidate.
            # We score each by (match_count * frac_within_target) / (1 + median_r_um).
            # ------------------------------------------------------------------
            if use_flex and (simple_candidate is not None) and bool(getattr(cfg, 'auto_flex_choose_best', True)):
                target_um = float(getattr(cfg, 'auto_flex_disp_target_um', 2.0))

                dx_p, dy_p = compute_registered_pair_displacements(cell_to_index_map, aligned_centroid_locations, ref_idx)
                q_p = displacement_quality(dx_p, dy_p, cfg.microns_per_pixel, target_um=target_um)
                q_s = simple_candidate.get('quality', {})

                def _score(q):
                    try:
                        mc = float(q.get('match_count', 0.0))
                    except Exception:
                        mc = 0.0
                    try:
                        frac = float(q.get('frac_within_target', 0.0))
                    except Exception:
                        frac = 0.0
                    try:
                        med = float(q.get('median_r_um', 1e9))
                    except Exception:
                        med = 1e9
                    if not np.isfinite(frac):
                        frac = 0.0
                    if not np.isfinite(med):
                        med = 1e9
                    return mc * frac / (1.0 + med)

                sp = _score(q_p)
                ss = _score(q_s)

                try:
                    flex_debug['prob_quality'] = q_p
                    flex_debug['prob_score'] = float(sp)
                    flex_debug['iou_score'] = float(ss)
                except Exception:
                    pass

                if ss > sp:
                    print(f"  ⚡ auto-flex selected IoU+Hungarian (score {ss:.3f} > {sp:.3f}).")
                    cell_to_index_map = simple_candidate['cell_to_index_map']
                    centroid_distance_map = simple_candidate['centroid_distance_map']
                    spatial_correlation_map = simple_candidate['iou_map']
                    registered_distances = simple_candidate['registered_distances']
                    non_registered_distances = simple_candidate['non_registered_distances']
                    model_used = 'Simple IoU+Hungarian (auto-flex)'
                    initial_metric_threshold = float(getattr(cfg, 'simple_iou_threshold', 0.25))
                    p_same_models = dict(auto_flex_overrode_probabilistic=True)
                    try:
                        flex_debug['selected'] = 'iou_hungarian'
                    except Exception:
                        pass
                else:
                    try:
                        flex_debug['selected'] = 'probabilistic'
                    except Exception:
                        pass

        # Step 6: Save results
        print("  Saving registration results...")
        
        results = {
            'cell_to_index_map': cell_to_index_map,
            'cell_to_index_map_pre_spatial_floor': pre_spatial_floor_map if 'pre_spatial_floor_map' in locals() else None,
            'dual_model_enabled': bool(use_dual_model),
            'spatial_corr_floor': float(spatial_corr_floor),
            'n_vetoed_clusters': int(n_vetoed_clusters) if 'n_vetoed_clusters' in locals() else 0,
            'vetoed_centroid_um': np.asarray(vetoed_centroid_um, dtype=float) if 'vetoed_centroid_um' in locals() else np.array([]),
            'vetoed_spatial_corr': np.asarray(vetoed_spatial_corr, dtype=float) if 'vetoed_spatial_corr' in locals() else np.array([]),
            'spatial_correlation_map': spatial_correlation_map,
            'registered_cells_correlations': registered_correlations,
            'non_registered_cells_correlations': non_registered_correlations,
            'centroid_distance_map': centroid_distance_map,
            'registered_cells_distances': registered_distances,
            'non_registered_cells_distances': non_registered_distances,
            'model_used': model_used,
            'initial_metric_threshold': float(initial_metric_threshold),
            'centroid_overlap_mse': float(centroid_overlap_mse),
            'corr_overlap_mse': float(corr_overlap_mse),
            'centroid_intersection': float(centroid_intersection),
            'corr_intersection': float(corr_intersection),
            'centroid_locations': aligned_centroid_locations,
            'session_paths': [str(s) for s in sessions],
            'mean_image_alignment': {
                'reference_session_index': int(ref_idx),
                'scores': maximal_cross_correlation,
                'translations': alignment_translations,
                'sources': alignment_sources,
                'raw_image_correlation': raw_image_correlation,
                'aligned_image_correlation': aligned_image_correlation,
            },
            'n_sessions': n_sessions,
            'p_same_models': p_same_models,
            'auto_flex': flex_debug if use_flex else None,
            'iou_candidate': simple_candidate if use_flex else None,
            'config': {
                'microns_per_pixel': cfg.microns_per_pixel,
                'maximal_distance': cfg.maximal_distance,
                'p_same_threshold': cfg.p_same_threshold,
                'model_type': cfg.model_type,
                'registration_approach': cfg.registration_approach,
            }
        }
        
        # Save as .npy
        np.save(results_dir / 'cell_to_index_map.npy', cell_to_index_map)
        if use_dual_model and 'pre_spatial_floor_map' in locals():
            try:
                np.save(results_dir / 'cell_to_index_map_pre_spatial_floor.npy', pre_spatial_floor_map)
            except Exception:
                pass
        np.save(results_dir / 'spatial_correlation_map.npy', spatial_correlation_map)
        np.save(results_dir / 'registration_results.npy', results)
        
        # Also save as .mat for MATLAB compatibility
        try:
            savemat(str(results_dir / 'cell_to_index_map.mat'),
                   {'cell_to_index_map': cell_to_index_map}, do_compression=True)
        except Exception as e:
            warnings.warn(f"Could not save .mat file: {e}")
        
        # ------------------------------------------------------------------
        # FIGURES (saved per-FOV; mirrors demo_validate_alignment / validation)
        # For the dual-model workflow we intentionally OMIT 'Stage 4' and 'Stage 5'
        # plots, and instead save a 'Stage 6 (dual)' summary including:
        #   - centroid distance histogram (accepted vs vetoed)
        #   - spatial correlation histogram (accepted vs vetoed, floor shown)
        #   - final registered projections + pairwise overlap
        # ------------------------------------------------------------------
        if bool(getattr(cfg, 'save_figures', False)):
            try:
                import os
                import matplotlib.pyplot as plt
                try:
                    from .plotting import (
                    plot_session_projections,
                    validate_alignment_deck,
                    plot_x_y_displacements,
                    plot_models,
                    plot_all_registered_projections,
                    plot_pairwise_session_overlap,
                    savefig_both,
                    _extract_p_from_model_string,
                    _compute_histogram_distribution,
                    )
                except Exception:
                    from cellregpy.plotting import (
                    plot_session_projections,
                    validate_alignment_deck,
                    plot_x_y_displacements,
                    plot_models,
                    plot_all_registered_projections,
                    plot_pairwise_session_overlap,
                    savefig_both,
                    _extract_p_from_model_string,
                    _compute_histogram_distribution,
                    )

                fig_dir = results_dir.parent / 'Figures'
                fig_dir.mkdir(parents=True, exist_ok=True)
                fig_dir_s = str(fig_dir)

                show_figs = str(getattr(cfg, 'figures_visibility', 'off')).strip().lower() == 'on'
                also_pdf = bool(getattr(cfg, 'also_pdf', False))
                close_figs = bool(getattr(cfg, 'close_figures', True))

                # Stage 1 — session projections (raw, adjusted footprints)
                plot_session_projections(footprint_projections, fig_dir_s, show=show_figs, also_pdf=also_pdf)

                # Stage 2 — alignment deck (uses aligned projections so we don't depend on dx/dy bookkeeping)
                aligned_proj = compute_footprint_projections(aligned_fps)
                validate_alignment_deck(
                    mean_images=fov_mean_images,
                    footprints_proj_raw=footprint_projections,
                    footprints_proj_aligned=aligned_proj,
                    reference_session_index=ref_idx,
                    alignment_translations=alignment_translations,
                    scores=maximal_cross_correlation,
                    out_dir=fig_dir_s,
                    session_names=[Path(s).parents[2].name if isinstance(s, (str, Path)) else str(s) for s in sessions],
                    show=show_figs,
                    also_pdf=also_pdf,
                )

                if not skip_probabilistic:

                    # Stage 3 — displacement + mixture models
                    plot_x_y_displacements(
                        data_dist['neighbors_x_displacements'],
                        data_dist['neighbors_y_displacements'],
                        cfg.microns_per_pixel,
                        max_dist_px,
                        number_of_bins,
                        centers_of_bins,
                        fig_dir_s,
                        show=show_figs,
                        also_pdf=also_pdf,
                    )

                    p_centroid = _extract_p_from_model_string(centroid_best_model)
                    p_spatial = _extract_p_from_model_string(corr_best_model)
                    centroid_dist_distribution = _compute_histogram_distribution(
                        data_dist['neighbors_centroid_distances'], centers_of_bins[0], number_of_bins, scale=cfg.microns_per_pixel
                    )
                    spatial_corr_distribution = _compute_histogram_distribution(
                        np.asarray(data_dist['neighbors_spatial_correlations']).ravel()[np.asarray(data_dist['neighbors_spatial_correlations']).ravel() >= 0],
                        centers_of_bins[1], number_of_bins, scale=1.0
                    )

                    plot_models(
                        np.array([p_centroid]),
                        data_dist['NN_centroid_distances'],
                        data_dist['NNN_centroid_distances'],
                        centroid_dist_distribution,
                        centroid_same_model,
                        centroid_diff_model,
                        centroid_mixture_model,
                        centroid_intersection,
                        centers_of_bins[0],
                        spatial_correlations_model_parameters=np.array([p_spatial]),
                        NN_spatial_correlations=data_dist['NN_spatial_correlations'],
                        NNN_spatial_correlations=data_dist['NNN_spatial_correlations'],
                        spatial_correlations_distribution=spatial_corr_distribution,
                        spatial_correlations_model_same_cells=corr_same_model,
                        spatial_correlations_model_different_cells=corr_diff_model,
                        spatial_correlations_model_weighted_sum=corr_mixture_model,
                        spatial_correlation_intersection=corr_intersection,
                        centers_of_bins_corr=centers_of_bins[1],
                        microns_per_pixel=cfg.microns_per_pixel,
                        maximal_distance=max_dist_px,
                        out_dir=fig_dir_s,
                        show=show_figs,
                        also_pdf=also_pdf,
                    )

                # Stage 6 (dual) — compute accepted pair metrics from FINAL map
                accepted_centroid_um = []
                accepted_spatial_corr = []
                try:
                    for cluster_idx in range(cell_to_index_map.shape[0]):
                        row = cell_to_index_map[cluster_idx, :]
                        present_sessions = np.where(row > 0)[0]
                        if present_sessions.size < 2:
                            continue
                        for ii in range(present_sessions.size):
                            si = int(present_sessions[ii])
                            ci = int(row[si]) - 1
                            for jj in range(ii + 1, present_sessions.size):
                                sj = int(present_sessions[jj])
                                cj = int(row[sj]) - 1
                                fp_i = aligned_fps[si][ci]
                                fp_j = aligned_fps[sj][cj]
                                sc = compute_spatial_correlation(fp_i, fp_j)
                                di = aligned_centroid_locations[si][ci]
                                dj = aligned_centroid_locations[sj][cj]
                                dpx = float(np.sqrt(np.sum((di - dj) ** 2)))
                                accepted_centroid_um.append(dpx * float(cfg.microns_per_pixel))
                                accepted_spatial_corr.append(float(sc))
                except Exception:
                    pass

                # Stage 6 histograms (centroid + spatial) — replaces Stage 4/5 plots
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                # centroid distances
                bins_d = np.linspace(0, float(cfg.maximal_distance), max(20, int(number_of_bins)))
                axes[0].hist(accepted_centroid_um, bins=bins_d, alpha=0.7, label='Accepted')
                axes[0].hist(vetoed_centroid_um, bins=bins_d, alpha=0.7, label='Vetoed')
                axes[0].set_xlabel('Centroid distance (µm)')
                axes[0].set_ylabel('Count')
                axes[0].legend(frameon=False)
                # spatial correlations
                bins_c = np.linspace(0, 1, max(20, int(number_of_bins)))
                axes[1].hist(accepted_spatial_corr, bins=bins_c, alpha=0.7, label='Accepted')
                axes[1].hist(vetoed_spatial_corr, bins=bins_c, alpha=0.7, label='Vetoed')
                axes[1].axvline(float(spatial_corr_floor), linestyle='--', linewidth=2)
                axes[1].set_xlabel('Spatial correlation')
                axes[1].set_ylabel('Count')
                axes[1].legend(frameon=False)
                fig.suptitle(f'Stage 6 (Dual) — centroid-primary + spatial floor={float(spatial_corr_floor):.2f}', fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                savefig_both(fig, os.path.join(fig_dir_s, 'Stage 6 - dual histograms'), also_pdf=also_pdf, show=show_figs)
                if close_figs and (not show_figs):
                    plt.close(fig)

                # Stage 6 projections + overlap (final map only)
                plot_all_registered_projections(aligned_fps, cell_to_index_map, fig_dir_s, show=show_figs, also_pdf=also_pdf, stage_label='Stage 6 (dual)')
                plot_pairwise_session_overlap(aligned_fps, cell_to_index_map, fig_dir_s, show=show_figs)

                # Stage 6b — registered-pair displacement cloud for the FINAL map (should be near 0,0 if matches are correct)
                try:
                    dx_f, dy_f = compute_registered_pair_displacements(cell_to_index_map, aligned_centroid_locations, ref_idx)
                    q_f = displacement_quality(dx_f, dy_f, cfg.microns_per_pixel, target_um=float(getattr(cfg, 'auto_flex_disp_target_um', 2.0)))
                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(111)
                    ax.scatter(dx_f * cfg.microns_per_pixel, dy_f * cfg.microns_per_pixel, s=6)
                    ax.axhline(0.0)
                    ax.axvline(0.0)
                    ax.set_xlabel("dx (µm)")
                    ax.set_ylabel("dy (µm)")
                    ax.set_title(f"Registered pair displacements (FINAL)\nN={int(q_f.get('match_count',0))} | med r={q_f.get('median_r_um', float('nan')):.2f} µm")
                    savefig_both(fig, os.path.join(fig_dir_s, "Stage6b_registered_displacements_FINAL"), also_pdf=also_pdf)
                    if close_figs and (not show_figs):
                        plt.close(fig)
                except Exception:
                    pass

                # Stage 7 — IoU+Hungarian candidate QC (auto-flex)
                if use_flex and (simple_candidate is not None):
                    try:
                        plot_all_registered_projections(aligned_fps, simple_candidate['cell_to_index_map'], fig_dir_s, show=show_figs, also_pdf=also_pdf, stage_label='Stage 7 (IoU+Hungarian candidate)')
                        plot_pairwise_session_overlap(aligned_fps, simple_candidate['cell_to_index_map'], fig_dir_s, show=show_figs)

                        dx_s, dy_s = compute_registered_pair_displacements(simple_candidate['cell_to_index_map'], aligned_centroid_locations, ref_idx)
                        q_s = simple_candidate.get('quality', {})
                        fig = plt.figure(figsize=(5, 5))
                        ax = fig.add_subplot(111)
                        ax.scatter(dx_s * cfg.microns_per_pixel, dy_s * cfg.microns_per_pixel, s=6)
                        ax.axhline(0.0)
                        ax.axvline(0.0)
                        ax.set_xlabel("dx (µm)")
                        ax.set_ylabel("dy (µm)")
                        ax.set_title(f"Registered pair displacements (IoU)\nN={int(q_s.get('match_count',0))} | med r={q_s.get('median_r_um', float('nan')):.2f} µm")
                        savefig_both(fig, os.path.join(fig_dir_s, "Stage7_registered_displacements_IOU"), also_pdf=also_pdf)
                        if close_figs and (not show_figs):
                            plt.close(fig)
                    except Exception:
                        pass

                # Stage 2b — auto-flex sweep summary (if available)
                if use_flex and isinstance(flex_debug, dict) and ('sweep' in flex_debug):
                    try:
                        rows = [r for r in flex_debug.get('sweep', []) if isinstance(r, dict) and ('error' not in r)]
                        if len(rows) > 0:
                            xs = [float(r.get('max_dist_um', float('nan'))) for r in rows]
                            fr = [float(r.get('frac_within_target', float('nan'))) for r in rows]
                            mr = [float(r.get('median_r_um', float('nan'))) for r in rows]
                            fig = plt.figure(figsize=(6, 4))
                            ax = fig.add_subplot(111)
                            ax.plot(xs, fr, marker='o')
                            ax.set_xlabel("maximal_distance (µm)")
                            ax.set_ylabel(f"frac within {float(flex_debug.get('target_um', 2.0)):.1f} µm")
                            ax2 = ax.twinx()
                            ax2.plot(xs, mr, marker='s')
                            ax2.set_ylabel("median r (µm)")
                            ax.set_title("Auto-flex sweep metrics")
                            savefig_both(fig, os.path.join(fig_dir_s, "Stage2b_auto_flex_sweep_metrics"), also_pdf=also_pdf)
                            if close_figs and (not show_figs):
                                plt.close(fig)
                    except Exception:
                        pass

                if close_figs and (not show_figs):
                    plt.close('all')

            except Exception as e:
                warnings.warn(f'Figure generation failed: {e}')
        print(f"  ✓ Registered {cell_to_index_map.shape[0]} cell clusters across {n_sessions} sessions")
    
    def _get_sessions_from_folder(self, mouse_folder: Path) -> List[Path]:
        """Get session paths from mouse folder."""
        return list_session_folders(mouse_folder)
       
    def _build_mouse_data(self, 
                         sessions: List[Path],
                         mouse_name: str,
                         skip_cellreg: bool = False) -> Dict:
        """
        Build mouse_data structure (minimal port needed for cellRegID propagation).

        This focuses on the pieces required to reproduce the MATLAB
        mouseDataCellTable() CellRegID assignment/propagation logic:
          - per-session iscell + paths
          - per-FOV CellReg results (cell_to_index_map + sessions_list)

        Notes:
          - This is intentionally lighter than full buildMouseData.m, which
            also merges behavior/ophys traces. You can extend later.
        """
        if sessions is None:
            sessions = []
        sessions = [Path(p) for p in sessions]

        if len(sessions) == 0:
            return {
                "mouse_name": mouse_name,
                "sessions": {},
                "CellReg": {},
            }

        # plane0_path is .../<session>/suite2p/plane0
        # mouse_folder is .../<mouse>
        try:
            mouse_folder = sessions[0].parents[2]
        except Exception:
            mouse_folder = sessions[0].parent

        mouse_data: Dict[str, Any] = {
            "mouse_name": mouse_name,
            "mouse_folder": str(mouse_folder),
            # sessions keyed by session_root path string
            "sessions": {},
            "CellReg": {},
        }

        # --- per-session minimal info ---
        for plane0_path in sessions:
            plane0_path = Path(plane0_path)
            # session_root = .../<session>
            session_root = plane0_path.parents[1] if plane0_path.name == "plane0" else plane0_path
            session_name = ensure_valid_field_name(session_root.name)
            session_key = session_name
            # ensure unique keys (rare, but can happen if folder names collide)
            if session_key in mouse_data["sessions"]:
                base = session_key
                c = 1
                while session_key in mouse_data["sessions"]:
                    c += 1
                    session_key = f"{base}_{c}"

            unix_time = get_session_unix_time(session_root)

            try:
                iscell = get_iscell(plane0_path)  # bool array length n_rois
            except Exception:
                iscell = None

            mouse_data["sessions"][session_key] = {
                "session_root": str(session_root),
                "session_name": session_root.name,
                "unixTime": unix_time,
                "plane0_path": str(plane0_path),
                "iscell": iscell,
                "n_rois": int(len(iscell)) if iscell is not None else None,
                "n_cells": int(np.sum(iscell)) if iscell is not None else None,
            }

        # --- per-FOV CellReg outputs (already computed by _register_fov) ---
        if not skip_cellreg:
            cellreg_root = mouse_folder / "1_CellReg"
            if cellreg_root.exists():
                cellreg_out: Dict[str, Any] = {}

                # FOV folders are named FOV1, FOV2, ...
                for fov_dir in sorted([p for p in cellreg_root.glob("FOV*") if p.is_dir()]):
                    results_dir = fov_dir / "Results"
                    map_path = results_dir / "cell_to_index_map.npy"
                    reg_path = results_dir / "registration_results.npy"

                    if not (map_path.exists() and reg_path.exists()):
                        continue

                    try:
                        cell_to_index_map = np.load(map_path, allow_pickle=True)
                    except Exception as e:
                        warnings.warn(f"Could not load {map_path}: {e}")
                        continue

                    try:
                        reg = np.load(reg_path, allow_pickle=True).item()
                    except Exception as e:
                        warnings.warn(f"Could not load {reg_path}: {e}")
                        reg = {}

                    sess_paths = reg.get("session_paths", None)
                    if sess_paths is None:
                        # fallbacks
                        sess_paths = reg.get("sessions_list", reg.get("sessions", []))

                    sess_paths = [str(s) for s in (sess_paths or [])]
                    if len(sess_paths) == 0:
                        # If missing, we can't map columns to sessions reliably
                        warnings.warn(f"{reg_path} missing session_paths; skipping {fov_dir.name}")
                        continue

                    # Convert mapping indices to suite2p ROI IDs when needed.
                    # MATLAB downstream expects suite2p ROI IDs.
                    cell_index_suite2p = np.array(cell_to_index_map, dtype=int, copy=True)

                    for col_i, sess_path in enumerate(sess_paths):
                        # sess_path is typically .../suite2p/plane0/CellReg.mat
                        try:
                            p = Path(sess_path)
                            plane0 = p.parent if p.suffix.lower() == ".mat" else p
                            iscell = get_iscell(plane0)
                        except Exception:
                            # If session not found, leave as-is
                            continue

                        n_rois = int(len(iscell))
                        s2p_ids = (np.where(iscell)[0] + 1).astype(int)  # suite2p ROI IDs for cells
                        n_cells = int(len(s2p_ids))

                        if n_cells == 0:
                            continue

                        col = cell_index_suite2p[:, col_i]
                        max_val = int(col.max()) if col.size else 0

                        # Heuristic:
                        # - If the mapping was produced in "cell-only index space" (1..n_cells),
                        #   max_val should be <= n_cells and n_cells < n_rois.
                        # - If it is already suite2p ROI IDs (1..n_rois), max_val can exceed n_cells.
                        if (n_cells < n_rois) and (max_val > 0) and (max_val <= n_cells):
                            mask = col > 0
                            valid = mask & (col <= n_cells)
                            mapped = np.zeros_like(col)
                            mapped[valid] = s2p_ids[col[valid] - 1]
                            cell_index_suite2p[:, col_i] = mapped

                    cellreg_out[fov_dir.name] = {
                        "cell_index": cell_index_suite2p,
                        "sessions_list": sess_paths,
                        "results_dir": str(results_dir),
                    }

                mouse_data["CellReg"] = cellreg_out

        return mouse_data

    def _build_cell_table(self, mouse_data: Dict) -> pd.DataFrame:
        """
        Build mouse_table DataFrame (CellReg-focused port) with **merge-aware** transitive closure.

        Compared to the earlier MATLAB-faithful "fill zeros only" propagation, this version additionally
        **reconciles/merges** labels when a later FOV indicates that two already-labeled groups are in
        fact the same cell.

        Implementation strategy:
          - Build a base table: one row per (SessionPath, suite2pID) where iscell==True
          - Treat each (SessionPath, suite2pID) as a node
          - Each CellReg FOV 'cell_index' row induces edges between all non-zero nodes in that row
          - Compute connected components across *all* FOVs (union-find)
          - Assign a unique cellRegID per component (order-independent transitive closure)
          - Track fovID membership as the set of FOVs that contributed evidence for the node/component
        """
        if mouse_data is None:
            return pd.DataFrame()

        mouse_name = mouse_data.get("mouse_name", "")
        sessions_dict: Dict[str, Any] = mouse_data.get("sessions", {}) or {}

        # ------------------------------------------------------------------ #
        # Build a base table: one row per (Session, suite2pID) for iscell==True
        # ------------------------------------------------------------------ #
        rows = []
        for session_key, sinfo in sessions_dict.items():
            iscell = sinfo.get("iscell", None)
            if iscell is None:
                continue
            iscell = np.asarray(iscell).astype(bool)
            s2p_ids = (np.where(iscell)[0] + 1).astype(int)

            for rid in s2p_ids:
                rows.append({
                    "MouseName": mouse_name,
                    "Session": str(sinfo.get("session_name", Path(session_key).name)),
                    "UnixTime": sinfo.get("unixTime", np.nan),
                    "suite2pID": int(rid),
                    "cellRegID": 0,
                    "fovID": [],   # list[int]
                    # extra (useful for joins back to disk paths)
                    "SessionPath": str(sinfo.get("session_root", session_key)),
                })

        mouse_table = pd.DataFrame(rows)
        if mouse_table.empty:
            return mouse_table

        # Fast lookups
        lookup = {(row.SessionPath, int(row.suite2pID)): int(i)
                  for i, row in mouse_table.iterrows()}

        rows_by_session: Dict[str, List[int]] = {}
        for i, row in mouse_table.iterrows():
            rows_by_session.setdefault(row["SessionPath"], []).append(int(i))

        def _session_root_from_cellreg_path(sess_path_str: str) -> str:
            p = Path(sess_path_str)
            plane0 = p.parent if p.suffix.lower() == ".mat" else p

            # Typical: .../<session>/suite2p/plane0
            if plane0.name == "plane0" and plane0.parent.name == "suite2p":
                return str(plane0.parent.parent)
            if plane0.name == "suite2p":
                return str(plane0.parent)

            # Fallback: search parents for suite2p
            for parent in [plane0] + list(plane0.parents):
                if parent.name == "suite2p":
                    return str(parent.parent)
            return str(plane0.parent)

        # Sort FOVs numerically when possible (FOV1, FOV2, ...)
        def _fov_sort_key(name: str):
            mm = re.search(r"(\d+)$", name)
            return int(mm.group(1)) if mm else 10**9

        cellreg_fovs: Dict[str, Any] = mouse_data.get("CellReg", {}) or {}
        if not cellreg_fovs:
            return mouse_table

        # ------------------------------------------------------------------ #
        # Union-Find over mouse_table rows (nodes)
        # ------------------------------------------------------------------ #
        n = len(mouse_table)
        parent = np.arange(n, dtype=int)
        rank = np.zeros(n, dtype=int)

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int):
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        touched = np.zeros(n, dtype=bool)  # nodes that appear in any cell_index
        fov_sets = [set() for _ in range(n)]  # evidence per node

        # Helper: map a CellReg session path into our SessionPath key space
        def _resolve_session_path(cellreg_sess_path: str) -> Optional[str]:
            session_path = _session_root_from_cellreg_path(cellreg_sess_path)
            if session_path in rows_by_session:
                return session_path

            # Fallback: match by basename (session folder name)
            base = Path(session_path).name
            candidate = next((k for k in rows_by_session.keys() if Path(k).name == base), None)
            return candidate

        # ------------------------------------------------------------------ #
        # Build edges from every FOV's cell_index rows and union them
        # ------------------------------------------------------------------ #
        for fov_name in sorted(cellreg_fovs.keys(), key=_fov_sort_key):
            fov = cellreg_fovs[fov_name]
            cellreg_index = fov.get("cell_index", None)
            sess_index = list(fov.get("sessions_list", []))

            if cellreg_index is None or len(sess_index) == 0:
                continue

            cellreg_index = np.asarray(cellreg_index).astype(int, copy=False)

            mm = re.search(r"(\d+)$", fov_name)
            fov_id = int(mm.group(1)) if mm else None

            # Pre-resolve session roots for this FOV's sessions (None if can't resolve)
            resolved_sess_roots: List[Optional[str]] = []
            for s in sess_index:
                resolved_sess_roots.append(_resolve_session_path(s))

            # For each row (a putative cell identity across sessions), union all nonzero nodes
            for r in range(cellreg_index.shape[0]):
                node_rows: List[int] = []
                for c, sess_root in enumerate(resolved_sess_roots):
                    if sess_root is None:
                        continue
                    sid = int(cellreg_index[r, c])
                    if sid <= 0:
                        continue
                    row_idx = lookup.get((sess_root, sid), None)
                    if row_idx is None:
                        continue
                    node_rows.append(int(row_idx))

                if not node_rows:
                    continue

                # Unique while preserving order
                node_rows = list(dict.fromkeys(node_rows))

                # Mark touched + fov evidence
                for idx in node_rows:
                    touched[idx] = True
                    if fov_id is not None:
                        fov_sets[idx].add(fov_id)

                # Union all nodes in this row
                base = node_rows[0]
                for other in node_rows[1:]:
                    _union(base, other)

        # ------------------------------------------------------------------ #
        # Assign cellRegID per connected component (order-independent)
        # Only components that contain at least one touched node get an ID.
        # ------------------------------------------------------------------ #
        # Collect components among touched nodes
        comp_members: Dict[int, List[int]] = {}
        for i in range(n):
            if not touched[i]:
                continue
            root = _find(i)
            comp_members.setdefault(root, []).append(i)

        # Assign IDs in a stable order (by smallest row index in component)
        cellreg_id = 0
        for root in sorted(comp_members.keys(), key=lambda r: min(comp_members[r])):
            cellreg_id += 1
            members = comp_members[root]

            # Aggregate fov evidence across the component and write back to each member
            comp_fovs: set[int] = set()
            for idx in members:
                comp_fovs |= fov_sets[idx]
            comp_fov_list = sorted(comp_fovs) if comp_fovs else []

            for idx in members:
                mouse_table.at[idx, "cellRegID"] = cellreg_id
                mouse_table.at[idx, "fovID"] = comp_fov_list

        return mouse_table

    def _save_results(self,
                     mouse_folder: Path,
                     mouse_data: Dict,
                     mouse_table: pd.DataFrame):
        """Save results to disk."""
        # Save as .npy for Python
        np.save(mouse_folder / 'mouse_data.npy', mouse_data)
        mouse_table.to_pickle(mouse_folder / 'mouse_table.pkl')
        
        # Also save as .mat for MATLAB compatibility
        # Save table even if mouse_data export fails
        try:
            table_dict = sanitize_for_mat(mouse_table.to_dict('list'))
            savemat(str(mouse_folder / 'mouse_table.mat'),
                   {'mouse_table': table_dict},
                   do_compression=True,
                   long_field_names=True)
        except Exception as e:
            warnings.warn(f"Could not save mouse_table.mat: {e}")

        # Save mouse_data with MATLAB-safe fieldnames
        try:
            mouse_data_mat = sanitize_for_mat(mouse_data)
            savemat(str(mouse_folder / 'mouse_data.mat'),
                   {'mouse_data': mouse_data_mat},
                   do_compression=True,
                   long_field_names=True)
        except Exception as e:
            warnings.warn(f"Could not save mouse_data.mat: {e}")


# ============================================================================ #
#                     SPATIAL FOOTPRINT PROCESSING                             #
# ============================================================================ #

def gaussfit(x: np.ndarray, y: np.ndarray, sigma_init: float = 5.0, 
             plot_flag: bool = False) -> Tuple[np.ndarray, float]:
    """
    Gaussian fit to data for centroid estimation.
    
    Mirrors MATLAB gaussfit.m.
    
    Args:
        x: x-values
        y: y-values (normalized to sum to 1)
        sigma_init: Initial sigma estimate
        plot_flag: Whether to plot the fit
        
    Returns:
        Tuple of (fitted y-values, centroid offset)
    """
    from scipy.optimize import curve_fit
    
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Initial guesses
    y = np.array(y, dtype=np.float64)
    y = y / (y.sum() + 1e-10)  # Normalize
    
    try:
        # Estimate initial parameters
        mu_init = np.sum(x * y)
        amp_init = y.max()
        
        popt, _ = curve_fit(gaussian, x, y, 
                           p0=[amp_init, mu_init, sigma_init],
                           maxfev=1000)
        
        y_fit = gaussian(x, *popt)
        centroid_offset = popt[1]  # mu
        
    except Exception:
        # Fallback to center of mass
        centroid_offset = np.sum(x * y) / (y.sum() + 1e-10)
        y_fit = y
    
    return y_fit, centroid_offset


def normalize_footprints(spatial_footprints: List[Union[Path, np.ndarray]],
                        write_path: Optional[Path] = None,
                        pixel_threshold: float = 0.0) -> List[Union[Path, np.ndarray]]:
    """
    Normalize spatial footprints to sum to 1.
    
    Mirrors MATLAB normalize_spatial_footprints.m.
    
    Args:
        spatial_footprints: List of footprint arrays or paths
        write_path: Optional path to write normalized footprints (memory efficient)
        pixel_threshold: Threshold for low signal pixels (0-1)
        
    Returns:
        List of normalized footprints (or paths if write_path provided)
    """
    print("Normalizing spatial footprints...")
    n_sessions = len(spatial_footprints)
    normalized = []
    
    for n in range(n_sessions):
        print(f"  Session {n + 1}/{n_sessions}")
        
        # Load footprints
        if isinstance(spatial_footprints[n], (Path, str)):
            fps = get_spatial_footprints(spatial_footprints[n])
        else:
            fps = spatial_footprints[n]
        
        fps = np.array(fps, dtype=np.float32)
        
        if pixel_threshold > 0:
            # Normalize by max first
            tmp_max = fps.max(axis=(1, 2), keepdims=True)
            tmp_max[tmp_max == 0] = 1  # Avoid division by zero
            temp_fp = fps / tmp_max
            temp_fp[temp_fp < pixel_threshold] = 0
            
            # Now normalize by sum
            tmp_sum = temp_fp.sum(axis=(1, 2), keepdims=True)
            tmp_sum[tmp_sum == 0] = 1
            temp_fp = temp_fp / tmp_sum
        else:
            # Just normalize by sum
            tmp_sum = np.nansum(fps, axis=(1, 2), keepdims=True)
            tmp_sum[tmp_sum == 0] = 1
            temp_fp = fps / tmp_sum
        
        if write_path is not None:
            # Save to disk
            save_path = Path(write_path) / f'normalized_spatial_footprints_{n}.npy'
            np.save(save_path, temp_fp)
            normalized.append(save_path)
        else:
            normalized.append(temp_fp)
    
    return normalized


def adjust_fov_size(spatial_footprints: List[Union[Path, np.ndarray]]
                   ) -> Tuple[List[np.ndarray], np.ndarray, int, int, np.ndarray]:
    """
    Adjust FOV sizes to be the same across all sessions.
    
    Mirrors MATLAB adjust_FOV_size.m.
    
    Args:
        spatial_footprints: List of footprint arrays or paths
        
    Returns:
        Tuple of:
            - adjusted_spatial_footprints: List of adjusted footprints
            - adjusted_fov: 3D array (n_sessions, y, x) marking valid FOV regions
            - adjusted_x_size: Common x size
            - adjusted_y_size: Common y size
            - adjustment_zero_padding: 2D array (2, n_sessions) with padding amounts
    """
    print("Adjusting FOV sizes...")
    n_sessions = len(spatial_footprints)
    
    # Find maximum FOV size
    adjusted_x_size = 0
    adjusted_y_size = 0
    fov_all = []
    
    for n in range(n_sessions):
        # Load footprints to get shape
        if isinstance(spatial_footprints[n], (Path, str)):
            fps = get_spatial_footprints(spatial_footprints[n])
        else:
            fps = spatial_footprints[n]
        
        n_cells, sz_y, sz_x = fps.shape
        fov_all.append(np.ones((sz_y, sz_x)))
        adjusted_x_size = max(adjusted_x_size, sz_x)
        adjusted_y_size = max(adjusted_y_size, sz_y)
    
    # Adjust all footprints
    adjusted_spatial_footprints = []
    adjusted_fov = np.zeros((adjusted_y_size, adjusted_x_size, n_sessions))
    adjustment_zero_padding = np.zeros((2, n_sessions))
    
    for n in range(n_sessions):
        # Adjust FOV mask
        adjusted_fov_temp = fov_all[n]
        new_fov = np.zeros((adjusted_y_size, adjusted_x_size))
        new_fov[:adjusted_fov_temp.shape[0], :adjusted_fov_temp.shape[1]] = adjusted_fov_temp
        adjusted_fov[:, :, n] = new_fov
        
        # Load and adjust footprints
        if isinstance(spatial_footprints[n], (Path, str)):
            fps = get_spatial_footprints(spatial_footprints[n])
        else:
            fps = spatial_footprints[n]
        
        n_cells, sz_y, sz_x = fps.shape
        adjusted_fps = np.zeros((n_cells, adjusted_y_size, adjusted_x_size))
        adjusted_fps[:, :sz_y, :sz_x] = fps
        adjusted_spatial_footprints.append(adjusted_fps)
        
        # Record padding
        adjustment_zero_padding[0, n] = adjusted_x_size - sz_x
        adjustment_zero_padding[1, n] = adjusted_y_size - sz_y
    
    return adjusted_spatial_footprints, adjusted_fov, adjusted_x_size, adjusted_y_size, adjustment_zero_padding


def compute_footprint_projections(spatial_footprints: List[np.ndarray],
                                  pixel_weight_threshold: float = 0.5) -> List[np.ndarray]:
    """
    Compute projection of all cells onto FOV for each session.
    
    Mirrors MATLAB compute_footprints_projections.m.
    
    Args:
        spatial_footprints: List of footprint arrays (n_cells, y, x)
        pixel_weight_threshold: Threshold for visualization (0-1)
        
    Returns:
        List of 2D projection images per session
    """
    print("Computing footprint projections...")
    projections = []
    
    for n, fps in enumerate(spatial_footprints):
        print(f"  Session {n + 1}/{len(spatial_footprints)}")
        
        # Normalize each cell by its max
        fps_max = fps.max(axis=(1, 2), keepdims=True)
        fps_max[fps_max == 0] = 1
        normalized_fps = fps / fps_max
        
        # Threshold
        normalized_fps[normalized_fps < pixel_weight_threshold] = 0
        
        # Sum across cells
        projection = np.nansum(normalized_fps, axis=0)
        projections.append(projection)
    
    return projections


def compute_centroids(spatial_footprints: List[np.ndarray],
                      microns_per_pixel: float = 2.0) -> List[np.ndarray]:
    """
    Compute centroid locations for all cells from spatial footprints.
    
    Mirrors MATLAB compute_centroid_locations.m.
    
    Args:
        spatial_footprints: List of footprint arrays (n_cells, y, x)
        microns_per_pixel: Pixel scaling factor
        
    Returns:
        List of centroid arrays (n_cells, 2) with (x, y) coordinates
    """
    print("Computing centroid locations...")
    
    typical_cell_size = 12  # micrometers
    normalized_cell_size = typical_cell_size / microns_per_pixel
    gaussian_radius = round(2 * normalized_cell_size)
    
    centroid_locations = []
    
    for n, fps in enumerate(spatial_footprints):
        print(f"  Session {n + 1}/{len(spatial_footprints)}")
        
        n_cells = fps.shape[0]
        centroids = np.zeros((n_cells, 2))
        
        for k in range(n_cells):
            temp_fp = fps[k]
            
            # X and Y projections
            x_proj = np.sum(temp_fp, axis=0)
            y_proj = np.sum(temp_fp, axis=1)
            
            max_x_ind = np.argmax(x_proj)
            max_y_ind = np.argmax(y_proj)
            
            # Extract localized projections with zero padding
            def get_localized_proj(proj, max_ind, radius):
                n = len(proj)
                if max_ind >= radius and max_ind <= n - radius - 1:
                    return proj[max_ind - radius:max_ind + radius + 1]
                elif max_ind < radius:
                    pad_size = radius - max_ind
                    return np.concatenate([np.zeros(pad_size), 
                                          proj[:max_ind + radius + 1]])
                else:
                    pad_size = radius - (n - 1 - max_ind)
                    return np.concatenate([proj[max_ind - radius:], 
                                          np.zeros(pad_size)])
            
            loc_x = get_localized_proj(x_proj, max_x_ind, gaussian_radius)
            loc_y = get_localized_proj(y_proj, max_y_ind, gaussian_radius)
            
            # Normalize
            loc_x = loc_x / (loc_x.sum() + 1e-10)
            loc_y = loc_y / (loc_y.sum() + 1e-10)
            
            # Gaussian fit for sub-pixel centroid
            x_range = np.arange(-gaussian_radius, gaussian_radius + 1)
            _, centroid_x_offset = gaussfit(x_range, loc_x, 0.5 * normalized_cell_size)
            _, centroid_y_offset = gaussfit(x_range, loc_y, 0.5 * normalized_cell_size)
            
            centroids[k, 0] = max_x_ind + centroid_x_offset
            centroids[k, 1] = max_y_ind + centroid_y_offset
        
        centroid_locations.append(centroids)
    
    return centroid_locations


def compute_centroid_projections(centroid_locations: List[np.ndarray],
                                 spatial_footprints: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute projection of cell centroids onto FOV.
    
    Mirrors MATLAB compute_centroids_projections.m.
    
    Args:
        centroid_locations: List of centroid arrays (n_cells, 2)
        spatial_footprints: List of footprint arrays (to get dimensions)
        
    Returns:
        List of centroid projection images
    """
    projections = []
    
    for n, (centroids, fps) in enumerate(zip(centroid_locations, spatial_footprints)):
        _, y_size, x_size = fps.shape
        proj = np.zeros((y_size, x_size))
        
        for cx, cy in centroids:
            # Mark centroid position
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < x_size and 0 <= iy < y_size:
                proj[iy, ix] = 1
        
        projections.append(proj)
    
    return projections


def make_alignment_image_from_footprints(spatial_footprints: np.ndarray,
                                         pixel_weight_threshold: float = 0.5,
                                         blur_sigma: float = 1.0) -> np.ndarray:
    """
    Build a single 2D alignment image from a stack of spatial footprints.

    This lets the mean-image aligner backend operate on spatial-footprint
    projections when mean-image alignment fails.

    Args:
        spatial_footprints: Array of shape (n_cells, y, x)
        pixel_weight_threshold: Per-cell normalized threshold before projection
        blur_sigma: Optional mild blur applied to the final projection

    Returns:
        2D float image suitable for MeanImageAligner.align()
    """
    fps = np.asarray(spatial_footprints, dtype=np.float32)
    if fps.ndim != 3:
        raise ValueError(f"Expected (n_cells, y, x) footprints, got shape {fps.shape}")

    if fps.shape[0] == 0:
        return np.zeros(fps.shape[1:], dtype=np.float32)

    fps_max = np.nanmax(fps, axis=(1, 2), keepdims=True)
    fps_max[~np.isfinite(fps_max)] = 0
    fps_max[fps_max == 0] = 1
    norm = fps / fps_max
    norm[~np.isfinite(norm)] = 0

    if pixel_weight_threshold is not None and pixel_weight_threshold > 0:
        norm[norm < float(pixel_weight_threshold)] = 0

    proj = np.nansum(norm, axis=0).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        proj = gaussian_filter(proj, blur_sigma)
    return proj


# ============================================================================ #
#                       PROBABILISTIC MODELING                                 #


def choose_best_model(centroid_overlap_mse: float,
                      corr_overlap_mse: float,
                      *,
                      centroid_intersection: Optional[float] = None,
                      corr_intersection: Optional[float] = None,
                      prefer: str = "Spatial correlation",
                      tie_rel_tol: float = 0.01) -> str:
    """Select which likelihood model to trust (MATLAB: choose_best_model).

    MATLAB calls choose_best_model(...) after fitting BOTH the centroid-distance model and
    the spatial-correlation model, then sets:
        initial_registration_type = best_model_string;
        model_type               = best_model_string;

    Here we pick the model with lower overlap MSE (lower is better). We also treat
    NaN/Inf intersections or MSEs as invalid.

    Args:
        centroid_overlap_mse: MSE from compute_centroid_distances_model_custom
        corr_overlap_mse:     MSE from compute_spatial_correlations_model
        centroid_intersection: p_same==0.5 threshold in *pixels* (optional)
        corr_intersection:     p_same==0.5 threshold in *correlation units* (optional)
        prefer: model to prefer in a near-tie
        tie_rel_tol: treat as tie if |a-b| <= tie_rel_tol * min(a,b)

    Returns:
        "Centroid distance" or "Spatial correlation"
    """
    def bad(x):
        return (x is None) or (not np.isfinite(float(x)))

    c_bad = bad(centroid_overlap_mse) or bad(centroid_intersection)
    r_bad = bad(corr_overlap_mse) or bad(corr_intersection)

    if c_bad and r_bad:
        return prefer
    if c_bad and not r_bad:
        return "Spatial correlation"
    if r_bad and not c_bad:
        return "Centroid distance"

    c = float(centroid_overlap_mse)
    r = float(corr_overlap_mse)

    denom = max(1e-12, min(c, r))
    if abs(c - r) <= tie_rel_tol * denom:
        return prefer

    return "Centroid distance" if c < r else "Spatial correlation"


def initial_registration_iou_hungarian(spatial_footprints: list[np.ndarray],
                                       centroid_locations: list[np.ndarray],
                                       *,
                                       reference_session_index: int,
                                       maximal_distance: float,
                                       mask_threshold: float = 0.20,
                                       iou_threshold: float = 0.25,
                                       cost_beta: float = 0.25
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic registration using IoU + Hungarian assignment.

    Intended as a fallback when sessions are already extremely similar, where
    probabilistic mixture modeling (CellReg) can become ill-posed.

    Strategy:
      - Anchor clusters to the reference session.
      - For each other session, compute candidate matches within maximal_distance.
      - Cost = (1 - IoU) + cost_beta * (dist / maximal_distance)
      - Solve one-to-one assignment via Hungarian algorithm.
      - Accept matches only if IoU >= iou_threshold AND dist <= maximal_distance.
      - Add unmatched cells as new clusters (present only in that session).

    Notes:
      - This is reference-anchored. Cells absent in the reference but shared across
        later sessions will not be merged across those later sessions in this mode.

    Returns:
      (cell_to_index_map, centroid_distance_map, iou_map, registered_dists, non_registered_dists)
      All maps are 1-indexed like MATLAB.
    """
    from scipy.optimize import linear_sum_assignment

    n_sessions = len(spatial_footprints)
    if n_sessions == 0:
        return (np.zeros((0, 0), dtype=int),
                np.zeros((0, 0), dtype=float),
                np.zeros((0, 0), dtype=float),
                np.array([]), np.array([]))

    ref_idx = int(reference_session_index)
    if not (0 <= ref_idx < n_sessions):
        raise ValueError(f"reference_session_index out of range: {ref_idx}")

    # Reference cells
    ref_fps = spatial_footprints[ref_idx]
    n_ref = 0 if ref_fps is None else int(ref_fps.shape[0])

    cell_to_index_map = np.zeros((n_ref, n_sessions), dtype=int)
    if n_ref > 0:
        cell_to_index_map[:, ref_idx] = np.arange(1, n_ref + 1)

    # Store per-cluster metrics vs each session (NaN where not applicable)
    centroid_distance_map = np.full((n_ref, n_sessions), np.nan, dtype=float)
    iou_map = np.full((n_ref, n_sessions), np.nan, dtype=float)
    if n_ref > 0:
        centroid_distance_map[:, ref_idx] = 0.0
        iou_map[:, ref_idx] = 1.0

    reg_dists = []
    nonreg_dists = []

    # Precompute reference masks + pixel indices for IoU
    def _roi_indices(roi2d: np.ndarray) -> np.ndarray:
        r = np.asarray(roi2d, dtype=float)
        mx = float(np.nanmax(r)) if r.size else 0.0
        if not np.isfinite(mx) or mx <= 0:
            return np.array([], dtype=np.int32)
        thr = mx * float(mask_threshold)
        idx = np.flatnonzero(r >= thr).astype(np.int32)
        return idx

    ref_idx_lists = []
    if n_ref > 0:
        for k in range(n_ref):
            ref_idx_lists.append(_roi_indices(ref_fps[k]))

    max_dist = float(maximal_distance)

    # Helper: IoU of sparse index lists
    def _iou(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        if idx_a.size == 0 or idx_b.size == 0:
            return 0.0
        inter = np.intersect1d(idx_a, idx_b, assume_unique=False).size
        union = idx_a.size + idx_b.size - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    # Reference centroids (pixels)
    ref_cents = centroid_locations[ref_idx]
    if ref_cents is None:
        ref_cents = np.zeros((n_ref, 2), dtype=float)

    for s in range(n_sessions):
        if s == ref_idx:
            continue

        mov_fps = spatial_footprints[s]
        mov_cents = centroid_locations[s]
        n_mov = 0 if mov_fps is None else int(mov_fps.shape[0])

        if n_mov == 0:
            continue

        mov_idx_lists = [_roi_indices(mov_fps[k]) for k in range(n_mov)]

        # Build cost matrix (n_ref x n_mov)
        big = 1e6
        cost = np.full((n_ref, n_mov), big, dtype=float)

        # Candidate gating by centroid distance
        if n_ref > 0 and ref_cents is not None and mov_cents is not None and len(ref_cents) and len(mov_cents):
            for j in range(n_mov):
                d = np.sqrt(np.sum((ref_cents - mov_cents[j]) ** 2, axis=1))
                cand = np.where(d <= max_dist)[0]
                if cand.size == 0:
                    continue
                for r in cand:
                    iou = _iou(ref_idx_lists[r], mov_idx_lists[j])
                    # Lower is better
                    cost[r, j] = (1.0 - iou) + float(cost_beta) * (float(d[r]) / max_dist)
        else:
            # No centroids: compute IoU for all pairs
            for r in range(n_ref):
                for j in range(n_mov):
                    iou = _iou(ref_idx_lists[r], mov_idx_lists[j])
                    cost[r, j] = (1.0 - iou)

        if n_ref == 0:
            # No reference cells: every cell becomes its own cluster
            extra = np.zeros((n_mov, n_sessions), dtype=int)
            extra[:, s] = np.arange(1, n_mov + 1)
            cell_to_index_map = np.vstack([cell_to_index_map, extra]) if cell_to_index_map.size else extra
            continue

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_ref = set()
        assigned_mov = set()

        for r, j in zip(row_ind.tolist(), col_ind.tolist()):
            if not np.isfinite(cost[r, j]) or cost[r, j] >= big:
                continue
            dist = float(np.sqrt(np.sum((ref_cents[r] - mov_cents[j]) ** 2))) if (ref_cents is not None and mov_cents is not None) else float('nan')
            iou = _iou(ref_idx_lists[r], mov_idx_lists[j])

            if (np.isfinite(dist) and dist <= max_dist) and (iou >= float(iou_threshold)):
                cell_to_index_map[r, s] = int(j + 1)  # 1-indexed
                centroid_distance_map[r, s] = dist
                iou_map[r, s] = iou
                reg_dists.append(dist)
                assigned_ref.add(r)
                assigned_mov.add(j)
            else:
                if np.isfinite(dist):
                    nonreg_dists.append(dist)

        # Add unmatched moving cells as new clusters
        unmatched = [j for j in range(n_mov) if j not in assigned_mov]
        if unmatched:
            extra = np.zeros((len(unmatched), n_sessions), dtype=int)
            extra[:, s] = np.array(unmatched, dtype=int) + 1
            cell_to_index_map = np.vstack([cell_to_index_map, extra])

            # Expand metric maps with NaN rows
            extra_nan = np.full((len(unmatched), n_sessions), np.nan, dtype=float)
            centroid_distance_map = np.vstack([centroid_distance_map, extra_nan])
            iou_map = np.vstack([iou_map, extra_nan])

    return (cell_to_index_map,
            centroid_distance_map,
            iou_map,
            np.asarray(reg_dists, dtype=float),
            np.asarray(nonreg_dists, dtype=float))

def initial_registration_centroid_distances_custom(centroid_locations: list[np.ndarray],
                                                   maximal_distance: float,
                                                   centroid_distance_threshold: float
                                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initial registration using centroid distances (MATLAB: initial_registration_centroid_distances_custom).

    Strategy (MATLAB-like):
      - Start with session 1 as the registered set.
      - For each subsequent session, for each new cell:
          * consider registered cells within maximal_distance
          * choose the closest candidate (minimum distance)
      - Greedy assignment in ascending distance to avoid collisions.
      - Add unassigned cells as new clusters.

    Args:
        centroid_locations: list of (n_cells,2) arrays, in the *aligned reference frame*
        maximal_distance: maximum candidate radius (pixels)
        centroid_distance_threshold: p_same==0.5 threshold (pixels). Smaller is "same".

    Returns:
        (cell_to_index_map, registered_dists, non_registered_dists, centroid_distance_map)
        where maps are 1-indexed like MATLAB.
    """
    n_sessions = len(centroid_locations)
    reg_cents = centroid_locations[0].copy()
    n_reg = reg_cents.shape[0]

    cell_to_index_map = np.zeros((n_reg, n_sessions), dtype=int)
    cell_to_index_map[:, 0] = np.arange(1, n_reg + 1)

    centroid_distance_map = np.zeros((n_reg, n_sessions), dtype=float)
    centroid_distance_map[:, 0] = 0.0  # distance to itself

    registered_dists = []
    non_registered_dists = []

    for s in range(1, n_sessions):
        new_cents = centroid_locations[s]
        n_new = new_cents.shape[0]
        if n_new == 0:
            continue

        best_dist = np.full(n_new, np.inf, dtype=float)
        best_reg_idx = np.full(n_new, -1, dtype=int)

        # For each new cell, find closest registered within maximal_distance
        for k in range(n_new):
            d = np.sqrt(np.sum((reg_cents - new_cents[k]) ** 2, axis=1))
            cand = np.where(d < maximal_distance)[0]
            if cand.size == 0:
                continue
            j = cand[np.argmin(d[cand])]
            best_dist[k] = d[j]
            best_reg_idx[k] = j

        # Eligible pairs: distance <= threshold
        pairs = np.where((best_reg_idx >= 0) & (best_dist <= centroid_distance_threshold))[0]
        order = np.argsort(best_dist[pairs])  # smallest distance first
        pairs = pairs[order]

        taken = np.zeros(n_reg, dtype=bool)
        assigned = np.zeros(n_new, dtype=bool)

        for k in pairs:
            r = best_reg_idx[k]
            if not taken[r]:
                taken[r] = True
                assigned[k] = True
                cell_to_index_map[r, s] = k + 1
                centroid_distance_map[r, s] = best_dist[k]
                registered_dists.append(best_dist[k])
            else:
                non_registered_dists.append(best_dist[k])

        # Add unassigned as new clusters
        new_cells = np.where(~assigned)[0]
        if new_cells.size:
            n_add = new_cells.size
            new_rows = np.zeros((n_add, n_sessions), dtype=int)
            new_rows[:, s] = new_cells + 1
            cell_to_index_map = np.vstack([cell_to_index_map, new_rows])

            new_dist_rows = np.zeros((n_add, n_sessions), dtype=float)
            centroid_distance_map = np.vstack([centroid_distance_map, new_dist_rows])

            reg_cents = np.vstack([reg_cents, new_cents[new_cells]])
            n_reg = reg_cents.shape[0]

        # subthreshold candidates
        low = (best_dist > centroid_distance_threshold) & (best_reg_idx >= 0) & np.isfinite(best_dist)
        non_registered_dists.extend(best_dist[low].tolist())

    return (cell_to_index_map,
            np.asarray(registered_dists, dtype=float),
            np.asarray(non_registered_dists, dtype=float),
            centroid_distance_map)
# ============================================================================ #

def estimate_num_bins(spatial_footprints: List[np.ndarray],
                      maximal_distance: float) -> Tuple[int, np.ndarray]:
    """
    Estimate number of bins for distance/correlation histograms.
    
    Mirrors MATLAB estimate_number_of_bins.m.
    
    Args:
        spatial_footprints: List of footprint arrays
        maximal_distance: Maximum distance in pixels
        
    Returns:
        Tuple of (number_of_bins, bin_centers)
    """
    # Estimate based on typical values
    n_bins = round(maximal_distance * 1.1)
    n_bins = max(50, min(200, n_bins))
    
    centers = np.linspace(0, 1, n_bins)
    
    return n_bins, centers


def compute_spatial_correlation(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Compute spatial correlation between two footprints.
    
    Mimics MATLAB corr2() - 2D correlation of entire images.
    
    Args:
        fp1: First footprint (2D array)
        fp2: Second footprint (2D array)
        
    Returns:
        Pearson correlation coefficient
    """
    # MATLAB corr2(A, B) computes correlation of full 2D arrays
    # equivalent to: corrcoef(A(:), B(:))
    fp1_flat = fp1.flatten().astype(np.float64)
    fp2_flat = fp2.flatten().astype(np.float64)
    
    # Check for empty or constant arrays
    if fp1_flat.std() < 1e-10 or fp2_flat.std() < 1e-10:
        return 0.0
    
    r = np.corrcoef(fp1_flat, fp2_flat)[0, 1]
    return r if np.isfinite(r) else 0.0


def compute_data_distribution(spatial_footprints: List[np.ndarray],
                              centroid_locations: List[np.ndarray],
                              maximal_distance: float) -> Dict[str, Any]:
    """
    Compute distributions of distances and correlations for neighboring cell pairs.
    
    Mirrors MATLAB compute_data_distribution.m.
    
    Args:
        spatial_footprints: List of footprint arrays (n_cells, y, x) per session
        centroid_locations: List of centroid arrays (n_cells, 2) per session
        maximal_distance: Maximum distance in pixels
        
    Returns:
        Dictionary containing:
            - all_to_all_indexes: Cell-pairs within maximal distance
            - all_to_all_spatial_correlations: Correlations for those pairs
            - all_to_all_centroid_distances: Distances for those pairs
            - neighbors_spatial_correlations: Flattened correlation vector
            - neighbors_centroid_distances: Flattened distance vector
            - NN_spatial_correlations: Nearest neighbor correlations
            - NNN_spatial_correlations: Non-nearest neighbor correlations
            - NN_centroid_distances: Nearest neighbor distances
            - NNN_centroid_distances: Non-nearest neighbor distances
    """
    print("Computing cell-pair similarity distributions...")
    
    n_sessions = len(spatial_footprints)
    
    # Initialize outputs
    all_to_all_indexes = [None] * n_sessions
    all_to_all_spatial_correlations = [None] * n_sessions
    all_to_all_centroid_distances = [None] * n_sessions
    
    neighbors_corrs = []
    neighbors_dists = []
    neighbors_x_disp = []
    neighbors_y_disp = []
    nn_corrs = []
    nnn_corrs = []
    nn_dists = []
    nnn_dists = []
    
    for n in range(n_sessions):
        print(f"  Session {n + 1}/{n_sessions}")
        
        new_fps = spatial_footprints[n]
        new_cents = centroid_locations[n]
        n_cells = new_fps.shape[0]
        
        # Initialize per-session storage
        sess_corrs = [[None] * n_sessions for _ in range(n_cells)]
        sess_dists = [[None] * n_sessions for _ in range(n_cells)]
        sess_idxs = [[None] * n_sessions for _ in range(n_cells)]
        
        sessions_to_compare = [s for s in range(n_sessions) if s != n]
        
        for m in sessions_to_compare:
            other_fps = spatial_footprints[m]
            other_cents = centroid_locations[m]
            
            for k in range(n_cells):
                # Get distances to all cells in other session
                centroid = new_cents[k]
                distances = np.sqrt(np.sum((other_cents - centroid) ** 2, axis=1))
                
                # Find cells within maximal distance
                nearby_idx = np.where(distances < maximal_distance)[0]
                
                if len(nearby_idx) == 0:
                    continue
                
                # Compute correlations
                this_fp = new_fps[k]
                corr_vec = []
                
                for idx in nearby_idx:
                    other_fp = other_fps[idx]

                    # compute correlation (keep invalid as 0.0, like your current behavior)
                    if this_fp.sum() == 0 or other_fp.sum() == 0:
                        r = 0.0
                    else:
                        r = compute_spatial_correlation(this_fp, other_fp)
                        if not np.isfinite(r):
                            r = 0.0

                    corr_vec.append(float(r))

                    # ALWAYS store into the global neighbor distributions
                    neighbors_corrs.append(float(r))
                    neighbors_dists.append(float(distances[idx]))
                    neighbors_x_disp.append(float(centroid[0] - other_cents[idx, 0]))
                    neighbors_y_disp.append(float(centroid[1] - other_cents[idx, 1]))
                
                corr_vec = np.array(corr_vec)
                nearby_dists = distances[nearby_idx]
                
                # Store all-to-all data
                sess_corrs[k][m] = corr_vec
                sess_dists[k][m] = nearby_dists
                sess_idxs[k][m] = nearby_idx + 1
                
                # Nearest neighbor (max correlation)
                if len(corr_vec) > 0:

                    # pick the nearest neighbor by centroid distance
                    i_nn = int(np.argmin(nearby_dists))

                    nn_dists.append(float(nearby_dists[i_nn]))
                    nn_corrs.append(float(corr_vec[i_nn]))

                    # "other neighbors" = all except the NN
                    mask_other = np.ones(len(nearby_dists), dtype=bool)
                    mask_other[i_nn] = False
                    nnn_dists.extend(nearby_dists[mask_other].tolist())
                    nnn_corrs.extend(corr_vec[mask_other].tolist())

        all_to_all_spatial_correlations[n] = sess_corrs
        all_to_all_centroid_distances[n] = sess_dists
        all_to_all_indexes[n] = sess_idxs
    
    # Convert to arrays and filter
    neighbors_corrs = np.array(neighbors_corrs)
    neighbors_dists = np.array(neighbors_dists)
    nn_corrs = np.array(nn_corrs)
    nn_dists = np.array(nn_dists)
    nnn_corrs = np.array(nnn_corrs) if nnn_corrs else np.array([])
    nnn_dists = np.array(nnn_dists) if nnn_dists else np.array([])
    
    # Warn if too many negative correlations
    if len(neighbors_corrs) > 0:
        neg_frac = (neighbors_corrs < 0).sum() / len(neighbors_corrs)
        if neg_frac > 0.05:
            warnings.warn(
                f"Large fraction ({neg_frac:.1%}) of negative correlations found. "
                "Check microns_per_pixel or reduce maximal_distance."
            )
    
    return {
        'all_to_all_indexes': all_to_all_indexes,
        'all_to_all_spatial_correlations': all_to_all_spatial_correlations,
        'all_to_all_centroid_distances': all_to_all_centroid_distances,
        'neighbors_spatial_correlations': neighbors_corrs,
        'neighbors_centroid_distances': neighbors_dists,
        'neighbors_x_displacements': np.array(neighbors_x_disp),
        'neighbors_y_displacements': np.array(neighbors_y_disp),
        'NN_spatial_correlations': nn_corrs,
        'NNN_spatial_correlations': nnn_corrs,
        'NN_centroid_distances': nn_dists,
        'NNN_centroid_distances': nnn_dists,
    }


def initial_registration_spatial_corr(spatial_footprints: List[np.ndarray],
                                       centroid_locations: List[np.ndarray],
                                       maximal_distance: float,
                                       spatial_correlation_threshold: float
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initial cell registration based on spatial correlations.
    
    Mirrors MATLAB initial_registration_spatial_correlations_custom.m.
    
    Args:
        spatial_footprints: List of footprint arrays (n_cells, y, x) per session
        centroid_locations: List of centroid arrays (n_cells, 2) per session
        maximal_distance: Maximum distance in pixels for candidates
        spatial_correlation_threshold: Minimum correlation to register
        
    Returns:
        Tuple of:
            - cell_to_index_map: (n_clusters, n_sessions) mapping array
            - registered_cells_corrs: Correlations for registered cell pairs
            - non_registered_cells_corrs: Correlations for non-registered pairs
            - spatial_correlation_map: Correlation values for each registration
    """
    print("Performing initial registration by spatial correlation...")
    
    n_sessions = len(spatial_footprints)
    
    # Initialize from session 1
    reg_stack = spatial_footprints[0].copy()
    reg_cents = centroid_locations[0].copy()
    n_reg = reg_stack.shape[0]
    
    cell_to_index_map = np.zeros((n_reg, n_sessions), dtype=int)
    cell_to_index_map[:, 0] = np.arange(1, n_reg + 1)  # 1-indexed like MATLAB
    
    spatial_correlation_map = np.zeros((n_reg, n_sessions))
    spatial_correlation_map[:, 0] = 1.0
    
    registered_corrs = []
    non_registered_corrs = []
    
    for s in range(1, n_sessions):
        print(f"  Session {s + 1}/{n_sessions}")
        
        new_stack = spatial_footprints[s]
        new_cents = centroid_locations[s]
        n_new = new_stack.shape[0]
        
        if n_new == 0:
            continue
        
        # Find candidates for each new cell
        best_corr = np.zeros(n_new)
        #best_reg_idx = np.zeros(n_new, dtype=int)
        best_reg_idx = -np.ones(n_new, dtype=int)
        
        for k in range(n_new):
            # Find registered cells within distance
            distances = np.sqrt(np.sum((reg_cents - new_cents[k]) ** 2, axis=1))
            candidates = np.where(distances < maximal_distance)[0]
            
            if len(candidates) == 0:
                continue
            
            # Compute correlations
            new_fp = new_stack[k]
            max_corr = -np.inf
            max_idx = -1
            
            for idx in candidates:
                reg_fp = reg_stack[idx]
                if new_fp.sum() > 0 and reg_fp.sum() > 0:
                    corr = compute_spatial_correlation(new_fp, reg_fp)
                    if corr > max_corr:
                        max_corr = corr
                        max_idx = idx
            
            if max_idx >= 0 and np.isfinite(max_corr):
                best_corr[k] = max_corr
                best_reg_idx[k] = max_idx
        
        # Greedy assignment by correlation (highest first)
        pairs = np.where((best_reg_idx >= 0) & 
                        (best_corr >= spatial_correlation_threshold))[0]
        order = np.argsort(best_corr[pairs])[::-1]
        pairs = pairs[order]
        
        taken = np.zeros(n_reg, dtype=bool)
        assigned = np.zeros(n_new, dtype=bool)
        
        for k in pairs:
            r = best_reg_idx[k]
            if not taken[r]:
                taken[r] = True
                assigned[k] = True
                cell_to_index_map[r, s] = k + 1  # 1-indexed
                spatial_correlation_map[r, s] = best_corr[k]
                registered_corrs.append(best_corr[k])
            else:
                non_registered_corrs.append(best_corr[k])
        
        # Add unassigned cells as new clusters
        new_cells = np.where(~assigned)[0]
        if len(new_cells) > 0:
            n_add = len(new_cells)
            
            # Expand maps
            new_rows = np.zeros((n_add, n_sessions), dtype=int)
            new_rows[:, s] = new_cells + 1  # 1-indexed
            cell_to_index_map = np.vstack([cell_to_index_map, new_rows])
            
            new_corr_rows = np.zeros((n_add, n_sessions))
            spatial_correlation_map = np.vstack([spatial_correlation_map, new_corr_rows])
            
            # Expand registered stack
            reg_stack = np.concatenate([reg_stack, new_stack[new_cells]], axis=0)
            reg_cents = np.vstack([reg_cents, new_cents[new_cells]])
            n_reg = reg_stack.shape[0]
        
        # Log sub-threshold matches
        low = (best_corr < spatial_correlation_threshold) & (best_reg_idx >= 0)
        non_registered_corrs.extend(best_corr[low].tolist())
    
    return (cell_to_index_map, 
            np.array(registered_corrs),
            np.array(non_registered_corrs),
            spatial_correlation_map)



def _weighted_beta_fit(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """
    Weighted MLE fit for Beta(a,b) on x in (0,1) with weights w >= 0.
    Used as a replacement for MATLAB estimate_beta_mixture_params().
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    # Clean / clamp to open interval
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[mask]
    w = w[mask]
    if x.size < 50:
        return 1.0, 10.0

    x = np.clip(x, eps, 1 - eps)
    w = w / (w.sum() + 1e-12)

    m = (w * x).sum()
    v = (w * (x - m) ** 2).sum()

    # Method-of-moments start (robustified)
    if v <= 1e-8 or m * (1 - m) <= v:
        a0, b0 = 1.0, 10.0
    else:
        t = m * (1 - m) / v - 1
        a0 = max(1e-3, m * t)
        b0 = max(1e-3, (1 - m) * t)

    from scipy.special import betaln
    from scipy.optimize import minimize

    def nll(log_ab):
        a = np.exp(log_ab[0])
        b = np.exp(log_ab[1])
        # -sum w * logpdf
        ll = (w * ((a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b))).sum()
        return -ll

    res = minimize(nll, np.log([a0, b0]), method="L-BFGS-B")
    a, b = np.exp(res.x[0]), np.exp(res.x[1])
    return float(a), float(b)


def _matlab_hist(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Replicate MATLAB's hist(data, centers) behavior.
    Creates bins centered at each center value, returning one count per center."""
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < 2:
        return np.array([len(data)])
    step = centers[1] - centers[0]
    edges = np.empty(len(centers) + 1)
    edges[0] = centers[0] - step / 2
    for i in range(1, len(centers)):
        edges[i] = (centers[i - 1] + centers[i]) / 2
    edges[-1] = centers[-1] + step / 2
    counts, _ = np.histogram(data, bins=edges)
    return counts.astype(np.float64)


def _estimate_beta_params_nr(assignments: np.ndarray,
                             data: np.ndarray,
                             maximal_distance: float = 1.0) -> Tuple[float, float]:
    """Port of MATLAB estimate_beta_mixture_params.m.
    Fits beta distribution parameters via Newton-Raphson with
    digamma/trigamma functions — matches MATLAB exactly."""
    from scipy.special import psi, polygamma

    w = np.asarray(assignments, dtype=np.float64)
    x = np.asarray(data, dtype=np.float64)
    x = np.clip(x, 1e-10, maximal_distance * 0.9999)
    sw = w.sum()
    if sw <= 0:
        return 1.0, 1.0

    # Sufficient statistics (MATLAB lines 11-12)
    g1 = float(np.sum(w * np.log(x / maximal_distance)) / sw)
    g2 = float(np.sum(w * np.log((maximal_distance - x) / maximal_distance)) / sw)

    # Moment-matching initialization (MATLAB lines 15-22)
    sample_mean = float(np.sum(w * x) / sw)
    sample_var = float(np.sum(w * (x - sample_mean) ** 2) / sw)
    xbar = sample_mean / maximal_distance
    ssq = sample_var / (maximal_distance ** 2)
    if ssq <= 0 or xbar <= 0 or xbar >= 1 or xbar * (1 - xbar) <= ssq:
        p, q = 1.0, 1.0
    else:
        factor = xbar * (1 - xbar) / ssq - 1
        p = max(xbar * factor, 1e-3)
        q = max((1 - xbar) * factor, 1e-3)

    # Newton-Raphson (MATLAB lines 26-32)
    for _ in range(100):
        try:
            psi_pq = float(psi(p + q))
            grad = np.array([psi(p) - psi_pq - g1,
                             psi(q) - psi_pq - g2])
            tri_pq = float(polygamma(1, p + q))
            hess = np.array([[polygamma(1, p) - tri_pq, -tri_pq],
                             [-tri_pq, polygamma(1, q) - tri_pq]])
            step_vec = np.linalg.solve(hess, grad)
            p -= float(step_vec[0])
            q -= float(step_vec[1])
            p = max(p, 1e-6)
            q = max(q, 1e-6)
        except Exception:
            break

    return float(p), float(q)


def compute_spatial_correlations_model(neighbors_spatial_correlations: np.ndarray,
                                      number_of_bins: int,
                                      centers_of_bins: Tuple[np.ndarray, np.ndarray]
                                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, float]:
    """
    Faithful port of MATLAB compute_spatial_correlations_model.m

    Fits a 2-component mixture in dissimilarity space (data = 1 - corr):
    - same-cell component: lognormal on (1 - corr)
    - different-cell component: beta on (1 - corr)

    Uses MATLAB's exact EM algorithm with weighted M-step.

    Returns:
        p_same_given_spatial_correlation, same_model, different_model,
        mixture_model, intersection_point, best_model_string, overlap_mse
    """
    from scipy.stats import lognorm as _lognorm, beta as _beta

    corr = np.asarray(neighbors_spatial_correlations, dtype=np.float64)
    corr = corr[np.isfinite(corr)]
    # MATLAB: neighbors_spatial_correlations(neighbors_spatial_correlations<0)=[];
    #         neighbors_spatial_correlations(neighbors_spatial_correlations>1)=[];
    corr = corr[(corr >= 0) & (corr <= 1)]
    if corr.size == 0:
        raise ValueError("neighbors_spatial_correlations is empty.")

    spatial_correlations_centers = np.asarray(centers_of_bins[1], dtype=np.float64)

    # MATLAB: data = 1 - neighbors_spatial_correlations
    data = 1.0 - corr
    data = np.clip(data, 1e-10, 1 - 1e-10)

    # ---- normalization denominator (MATLAB style) ----
    step = spatial_correlations_centers[1] - spatial_correlations_centers[0]
    rng  = spatial_correlations_centers[-1] - spatial_correlations_centers[0]
    norm_denom = step + rng

    # ---- lognormal init from high-correlation data (MATLAB line 32) ----
    hi = data[corr >= 0.7]
    if hi.size >= 5:
        mu = float(np.mean(np.log(hi)))
        sigma = float(max(np.std(np.log(hi), ddof=1), 1e-6))
    else:
        mu, sigma = -4.0, 0.5

    # ---- beta init from low-correlation data (MATLAB line 34) ----
    # MATLAB: betafit(neighbors_spatial_correlations(neighbors_spatial_correlations<0.75))
    # betafit fits from corr values; then beta is applied to data=1-corr in EM.
    # Use Newton-Raphson matching MATLAB betafit:
    lo_corr = corr[corr < 0.75]
    lo_corr = np.clip(lo_corr, 1e-6, 1 - 1e-6)
    if lo_corr.size >= 5:
        p_beta, q_beta = _estimate_beta_params_nr(
            np.ones(lo_corr.size), lo_corr, maximal_distance=1.0)
    else:
        p_beta, q_beta = 1.0, 10.0

    # MATLAB: PIsame = 0.5
    PIsame = 0.5

    # ---- EM Algorithm (MATLAB lines 39-57) ----
    for _ in range(100):
        # E-step
        same_eval = _lognorm.pdf(data, s=max(sigma, 1e-6), scale=np.exp(mu))
        diff_eval = _beta.pdf(data, a=max(p_beta, 1e-6), b=max(q_beta, 1e-6))
        numer = PIsame * same_eval
        denom_em = numer + (1 - PIsame) * diff_eval + 1e-12
        assignments = numer / denom_em

        # M-step (MATLAB weighted formulas, lines 47-53)
        sum_assign = float(np.sum(assignments))
        if sum_assign > 0:
            PIsame = sum_assign / len(assignments)
            mu = float(np.sum(assignments * np.log(data)) / sum_assign)
            sigma = float(np.sqrt(
                np.sum(assignments * (np.log(data) - mu) ** 2) / sum_assign))
            sigma = max(sigma, 1e-6)

        # Beta M-step: MATLAB's estimate_beta_mixture_params (Newton-Raphson)
        p_beta, q_beta = _estimate_beta_params_nr(
            1 - assignments, data, maximal_distance=1.0)

    # ---- Evaluate models on correlation centers (MATLAB lines 55-56) ----
    x_cent = np.clip(1.0 - spatial_correlations_centers, 1e-10, 1 - 1e-10)
    same_model = _lognorm.pdf(x_cent, s=max(sigma, 1e-6), scale=np.exp(mu))
    different_model = _beta.pdf(x_cent, a=max(p_beta, 1e-6), b=max(q_beta, 1e-6))

    # ---- Normalize models (MATLAB lines 90-91) ----
    def _norm_pdf(pdf):
        s = pdf.sum()
        if s > 0:
            return pdf / s * (number_of_bins / norm_denom)
        return pdf

    same_model = _norm_pdf(same_model)
    different_model = _norm_pdf(different_model)

    # ---- Swap check: same_model must peak at HIGH correlation ----
    # If EM converged with swapped labels, fix it.
    same_peak_corr = float(spatial_correlations_centers[np.argmax(same_model)])
    diff_peak_corr = float(spatial_correlations_centers[np.argmax(different_model)])
    if same_peak_corr < diff_peak_corr:
        same_model, different_model = different_model, same_model
        PIsame = 1.0 - PIsame

    # ---- Sigmoid smoothing on same_model (MATLAB lines 95-98) ----
    def _sigmoid(x, a_, c_):
        return 1.0 / (1.0 + np.exp(-a_ * (x - c_)))

    smoothing = _sigmoid(spatial_correlations_centers,
                         20.0,
                         float(spatial_correlations_centers.min()) + 0.5)
    same_model = same_model * smoothing
    step_10 = max(1, round(number_of_bins / 10))
    same_model[::step_10] = 0

    # ---- Weighted sum (MATLAB line 102) ----
    mixture_model = PIsame * same_model + (1 - PIsame) * different_model

    # ---- Histogram distribution (MATLAB line 106-107) ----
    counts = _matlab_hist(corr, spatial_correlations_centers)
    total = max(counts.sum(), 1.0)
    distribution = counts / total * (number_of_bins / norm_denom)

    # ---- MSE (MATLAB line 110) ----
    overlap_mse = float(np.sum(np.abs(
        (distribution - mixture_model) * norm_denom / number_of_bins)) / 2)

    # ---- p_same(corr) (MATLAB lines 114-123) ----
    p_same_given_spatial_correlation = (PIsame * same_model) / (
        PIsame * same_model + (1 - PIsame) * different_model + 1e-12)

    # Sigmoid smoothing on p_same at low-same-model region (MATLAB lines 120-123)
    minimal_p_same_threshold = 0.001
    low_mask = same_model < minimal_p_same_threshold * max(same_model.max(), 1e-12)
    indexes_to_smooth = np.where(low_mask)[0]
    if indexes_to_smooth.size > 0:
        n_smooth = indexes_to_smooth.size
        smooth_x = np.arange(1, n_smooth + 1, dtype=np.float64)
        smooth_func = _sigmoid(smooth_x, 0.05 * n_smooth, 0.8 * n_smooth)
        p_same_given_spatial_correlation[indexes_to_smooth] *= smooth_func

    # ---- Intersection (MATLAB lines 126-134) ----
    above_thresh = np.where(
        same_model > minimal_p_same_threshold * max(same_model.max(), 1e-12))[0]
    if above_thresh.size > 1:
        search_range = above_thresh[:-1]  # MATLAB: index_range_of_intersection(end)=[]
        diffs = np.abs(PIsame * same_model[search_range] -
                       (1 - PIsame) * different_model[search_range])
        ii = int(np.argmin(diffs))
        intersection_point = round(
            spatial_correlations_centers[search_range[ii]] * 100) / 100
    else:
        # Fallback: search full range
        mask = (spatial_correlations_centers > 0.1) & (spatial_correlations_centers < 0.8)
        if mask.any():
            diffs = np.abs(PIsame * same_model[mask] -
                           (1 - PIsame) * different_model[mask])
            ii = int(np.argmin(diffs))
            intersection_point = float(spatial_correlations_centers[mask][ii])
        else:
            intersection_point = 0.5

    best_model_string = (f"p={PIsame:.3f}, logn(mu={mu:.3f},sigma={sigma:.3f}), "
                         f"beta(a={p_beta:.3f},b={q_beta:.3f})")
    return (p_same_given_spatial_correlation,
            same_model,
            different_model,
            mixture_model,
            intersection_point,
            best_model_string,
            overlap_mse)


def compute_centroid_distances_model_custom(neighbors_centroid_distances: np.ndarray,
                                           number_of_bins: int,
                                           centers_of_bins: Tuple[np.ndarray, np.ndarray],
                                           microns_per_pixel: float = 1.0,
                                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str, float]:
    """
    Faithful port of MATLAB compute_centroid_distances_model_custom.m

    Fits distance distribution in **microns**:
    - same-cell component: lognormal PDF
    - different-cell component: b*x / (1 + exp(-a*(x - c)))

    Uses joint 6-parameter fit [p, mu, sigma, a, c, b] via least_squares,
    matching MATLAB's lsqcurvefit approach.

    Returns:
        p_same_given_centroid_distance, same_model, different_model,
        mixture_model, intersection_point, best_model_string, overlap_mse
    """
    from scipy.stats import lognorm as _lognorm
    from scipy.optimize import curve_fit as _curve_fit
    from scipy.special import expit

    d = np.asarray(neighbors_centroid_distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("neighbors_centroid_distances is empty.")

    centers = np.asarray(centers_of_bins[0], dtype=np.float64)  # pixels
    mpp = float(microns_per_pixel)
    xdata = mpp * centers                                        # microns
    xdata[xdata <= 0] = np.finfo(float).eps

    # ---- histogram (MATLAB-style: hist(d, centers)) ----
    counts = _matlab_hist(d, centers)
    step  = mpp * (centers[1] - centers[0])
    rng   = mpp * (centers[-1] - centers[0])
    denom = step + rng
    total = max(counts.sum(), 1.0)
    distribution = counts / total * (number_of_bins / denom)

    # ---- initial parameters (MATLAB lines 19-51) ----
    max_fit_um = 9.0

    d_px = d

    # MATLAB: sel = microns_per_pixel*neighbors_centroid_distances < max_fit_um & ... > 0;
    #         d_fit = neighbors_centroid_distances(sel);  <-- pixel space!
    #         ph = lognfit(d_fit);  <-- lognfit on PIXELS
    sel = (mpp * d_px < max_fit_um) & (d_px > 0)
    d_fit = d_px[sel]
    if d_fit.size < 10:
        d_fit = d_px[d_px > 0]

    if d_fit.size > 0:
        try:
            # MATLAB lognfit returns MLE [mu, sigma] with ddof=1
            mu0 = float(np.mean(np.log(d_fit)))
            sg0 = float(max(np.std(np.log(d_fit), ddof=1), 1e-3))
        except Exception:
            mu0, sg0 = float(np.log(np.mean(xdata[xdata > 0]))), 0.5
    else:
        mu0, sg0 = float(np.log(np.mean(xdata[xdata > 0]))), 0.5
    p0 = min(max(d_fit.size / max(d_px.size, 1), 0.05), 0.95)

    # diff-shape initial guess
    mid = max(1, number_of_bins // 2)
    den_init = mpp * (centers[-1] - centers[mid]) + np.finfo(float).eps
    b0 = (distribution[-1] - distribution[mid]) / den_init
    b0 = max(b0 / (1 - p0), 1e-6)
    a0 = 1.0
    c0 = max(float(np.median(xdata)), 1.0)

    x0_vec = [p0, mu0, sg0, a0, c0, b0]

    # ---- mixture model function (MATLAB line 59-61) ----
    # curve_fit signature: f(xdata, *params) -> ydata
    def _mixture(xd, p_, mu_, sg_, a_, c_, b_):
        sg_ = max(sg_, 1e-6)
        same = (1.0 / (xd * sg_ * np.sqrt(2 * np.pi))) * np.exp(
            -(np.log(xd) - mu_) ** 2 / (2 * sg_ ** 2))
        #diff = b_ * xd / (1.0 + np.exp(-a_ * (xd - c_)))
        diff = b_ * xd * expit(a_ * (xd - c_))
        #diff_raw = b * xdata * expit(a * (xdata - c))

        return p_ * same + (1 - p_) * diff

    # ---- joint fit (MATLAB lsqcurvefit with Levenberg-Marquardt) ----
    # MATLAB uses 'levenberg-marquardt' which ignores bounds.
    # Use scipy curve_fit with method='lm' (no bounds) to match.
    # ---- joint fit (use bounded solver to avoid insane params / exp overflow) ----
    lb = np.array([0.0, -np.inf, 1e-6, 0.0, 0.0, 0.0])
    ub = np.array([1.0,  np.inf, 10.0, 1e3, np.max(xdata) * 10, np.inf])

    popt, _ = _curve_fit(
        _mixture,
        xdata,
        distribution,
        p0=x0_vec,
        bounds=(lb, ub),      # <-- this is the key change
        method='trf',         # <-- this replaces method='lm'
        maxfev=20000
    )
    p, mu, sg, a, c, b = popt

    # Clamp p to [0,1] and sg > 0 since LM doesn't enforce bounds
    p = float(np.clip(p, 0.0, 1.0))
    sg = max(sg, 1e-6)

    # ---- component PDFs (individually normalized like MATLAB lines 79-87) ----
    same_raw = _lognorm.pdf(xdata, s=sg, scale=np.exp(mu))
    #diff_raw = b * xdata / (1.0 + np.exp(-a * (xdata - c)))
    diff_raw = b * xdata * expit(a * (xdata - c))

    def _norm_pdf(pdf):
        s = pdf.sum()
        if s > 0:
            return pdf / s * (number_of_bins / denom)
        return pdf

    same_model = _norm_pdf(same_raw)
    different_model = _norm_pdf(diff_raw)
    mixture_model = p * same_model + (1 - p) * different_model


    # ---- p_same(d) ----
    p_same_given_centroid_distance = (p * same_model) / (
        p * same_model + (1 - p) * different_model + 1e-12)
    if len(p_same_given_centroid_distance) >= 2:
        p_same_given_centroid_distance[0] = p_same_given_centroid_distance[1]

    # ---- intersection (MATLAB lines 93-96) ----
    idx_rng = np.where((centers > 1.0 / mpp) & (centers < 10.0 / mpp))[0]
    if idx_rng.size > 0:
        diffs = np.abs(p * same_model[idx_rng] - (1 - p) * different_model[idx_rng])
        ii = int(np.argmin(diffs))
        intersection_point = round(mpp * centers[idx_rng[ii]] * 100) / 100
    else:
        intersection_point = float(
            mpp * centers[np.argmin(np.abs(p * same_model - (1 - p) * different_model))])

    # ---- MSE (MATLAB line 99-101) ----
    overlap_mse = float(np.sum(np.abs(
        (distribution - mixture_model) * denom / number_of_bins)) / 2)

    best_model_string = (f"p={p:.3f}, logn(mu={mu:.3f},sigma={sg:.3f}), "
                         f"diff(a={a:.3f},c={c:.3f},b={b:.3f})")
    return (p_same_given_centroid_distance,
            same_model,
            different_model,
            mixture_model,
            intersection_point,
            best_model_string,
            overlap_mse)


def compute_p_same(all_to_all_centroid_distances: List,
                   all_to_all_spatial_correlations: List,
                   centers_of_bins: Tuple[np.ndarray, np.ndarray],
                   p_same_given_centroid_distance: np.ndarray,
                   p_same_given_spatial_correlation: np.ndarray
                   ) -> Tuple[List, List]:
    """
    Port of MATLAB compute_p_same.m

    Returns:
      p_same_centroid_distances: nested cell-like list structure mirroring all_to_all_centroid_distances
      p_same_spatial_correlations: nested structure mirroring all_to_all_spatial_correlations
    """
    centroid_centers = np.asarray(centers_of_bins[0], dtype=np.float64)
    corr_centers = np.asarray(centers_of_bins[1], dtype=np.float64)

    n_sessions = len(all_to_all_centroid_distances)
    p_same_centroid_distances = [None] * n_sessions
    p_same_spatial_correlations = [None] * n_sessions

    for n in range(n_sessions):
        sess_d = all_to_all_centroid_distances[n]
        sess_c = all_to_all_spatial_correlations[n]

        # number of cells in this session is len(sess_d)
        n_cells = len(sess_d)
        out_d = [[None] * n_sessions for _ in range(n_cells)]
        out_c = [[None] * n_sessions for _ in range(n_cells)]

        for ind in range(n_cells):
            for other in range(n_sessions):
                temp_dist = sess_d[ind][other]
                if temp_dist is None:
                    continue
                temp_corr = sess_c[ind][other]
                if temp_corr is None:
                    continue

                td = np.asarray(temp_dist, dtype=np.float64)
                tc = np.asarray(temp_corr, dtype=np.float64)

                # Map each value to nearest bin center (MATLAB min(abs(...)))
                # Centroid distances
                bin_ids_d = np.abs(td[:, None] - centroid_centers[None, :]).argmin(axis=1)
                out_d[ind][other] = p_same_given_centroid_distance[bin_ids_d]

                # Spatial correlations
                bin_ids_c = np.abs(tc[:, None] - corr_centers[None, :]).argmin(axis=1)
                out_c[ind][other] = p_same_given_spatial_correlation[bin_ids_c]

        p_same_centroid_distances[n] = out_d
        p_same_spatial_correlations[n] = out_c

    return p_same_centroid_distances, p_same_spatial_correlations


def estimate_registration_accuracy(cell_to_index_map: np.ndarray,
                                  all_to_all_p_same: List,
                                  all_to_all_indexes: List,
                                  threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of MATLAB estimate_registration_accuracy.m

    Returns:
      p_same_vec: p_same for each registered pair
      p_different_vec: 1 - p_same for each registered pair
      scores: 3-element vector [mean(p_same), mean(1-p_same), mean(p_same)/(mean(p_same)+mean(1-p_same))]
    """
    n_sessions = cell_to_index_map.shape[1]
    p_same_vec = []
    p_diff_vec = []

    # For each session, iterate over registered cells and compare to other sessions
    for n in range(n_sessions):
        # number of cells in session n is max index present
        max_cell = int(cell_to_index_map[:, n].max())
        if max_cell <= 0:
            continue
        sessions_to_compare = [s for s in range(n_sessions) if s != n]

        for cell_i in range(1, max_cell + 1):
            inds = np.where(cell_to_index_map[:, n] == cell_i)[0]
            if inds.size == 0:
                continue
            ind = int(inds[0])

            for other in sessions_to_compare:
                mapped = int(cell_to_index_map[ind, other])
                if mapped <= 0:
                    continue

                # look up p_same for this pair from candidate list
                cand = all_to_all_indexes[n][cell_i - 1][other]
                ps = all_to_all_p_same[n][cell_i - 1][other]
                if cand is None or ps is None:
                    continue
                cand = np.asarray(cand, dtype=int)
                ps = np.asarray(ps, dtype=np.float64)
                j = np.where(cand == mapped)[0]
                if j.size == 0:
                    continue
                v = float(ps[int(j[0])])
                p_same_vec.append(v)
                p_diff_vec.append(1.0 - v)

    p_same_vec = np.asarray(p_same_vec, dtype=np.float64)
    p_diff_vec = np.asarray(p_diff_vec, dtype=np.float64)

    if p_same_vec.size == 0:
        scores = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    else:
        mps = float(np.nanmean(p_same_vec))
        mpd = float(np.nanmean(p_diff_vec))
        scores = np.array([mps, mpd, mps / (mps + mpd + 1e-12)], dtype=np.float64)

    return p_same_vec, p_diff_vec, scores


import numpy as np

def transform_distance_to_similarity(measured_distance: float, maximal_distance: float) -> float:
    return (maximal_distance - measured_distance) / maximal_distance


def compute_scores_matlab(cell_to_index_map: np.ndarray,
                          all_to_all_indexes,
                          all_to_all_p_same,
                          number_of_sessions: int):
    """
    Faithful port of CellReg compute_scores.m
    """
    n_clusters = cell_to_index_map.shape[0]
    cell_scores = np.full(n_clusters, np.nan, dtype=float)
    cell_scores_positive = np.full(n_clusters, np.nan, dtype=float)
    cell_scores_negative = np.full(n_clusters, np.nan, dtype=float)
    cell_scores_exclusive = np.full(n_clusters, np.nan, dtype=float)

    p_same_registered_pairs = []
    for _ in range(n_clusters):
        p_same_registered_pairs.append(np.full((number_of_sessions, number_of_sessions), np.nan, dtype=float))

    def _get_candidates(sess_a, cell_a_1idx, sess_b):
        if cell_a_1idx <= 0:
            return None, None
        cand = all_to_all_indexes[sess_a][cell_a_1idx - 1][sess_b]
        ps = all_to_all_p_same[sess_a][cell_a_1idx - 1][sess_b]
        if cand is None or ps is None:
            return None, None
        cand = np.asarray(cand, dtype=int).ravel()
        ps = np.asarray(ps, dtype=float).ravel()
        return cand, ps

    for n in range(n_clusters):
        good_pairs = 0.0
        good_pairs_positive = 0.0
        good_pairs_negative = 0.0
        good_pairs_exclusive = 0.0

        num_comp = 0
        num_comp_pos = 0
        num_comp_neg = 0

        cells_in_cluster = cell_to_index_map[n, :]

        for m in range(number_of_sessions):
            for k in range(number_of_sessions):
                if k == m:
                    continue
                if cells_in_cluster[m] <= 0:
                    continue

                this_cell = int(cell_to_index_map[n, m])
                num_comp += 1

                cand, ps = _get_candidates(m, this_cell, k)

                if cell_to_index_map[n, k] == 0:
                    # active-inactive (negative)
                    num_comp_neg += 1
                    if cand is None or ps is None or ps.size == 0:
                        good_pairs += 1.0
                        good_pairs_negative += 1.0
                    else:
                        good_pairs += 1.0 - float(np.sum(ps))
                        good_pairs_negative += 1.0 - float(np.sum(ps))

                else:
                    # active-active (positive)
                    num_comp_pos += 1
                    clustered_cell = int(cell_to_index_map[n, k])

                    temp_true_positive = 0.0
                    if cand is not None and ps is not None and ps.size > 0:
                        hit = np.where(cand == clustered_cell)[0]
                        if hit.size > 0:
                            temp_true_positive = float(ps[hit[0]])
                            p_same_registered_pairs[n][m, k] = temp_true_positive
                            # remove clustered entry
                            ps_rest = np.delete(ps, hit[0])
                        else:
                            ps_rest = ps
                    else:
                        ps_rest = np.array([], dtype=float)

                    good_pairs_positive += temp_true_positive

                    if ps_rest.size == 0:
                        good_pairs += temp_true_positive
                        good_pairs_exclusive += 1.0
                    else:
                        good_pairs += temp_true_positive - float(np.sum(ps_rest))
                        good_pairs_exclusive += 1.0 - float(np.sum(ps_rest))

        if num_comp_pos > 0:
            cell_scores_positive[n] = good_pairs_positive / num_comp_pos
            cell_scores_exclusive[n] = good_pairs_exclusive / num_comp_pos

        if num_comp_neg > 0:
            cell_scores_negative[n] = good_pairs_negative / num_comp_neg

        if num_comp > 0:
            cell_scores[n] = good_pairs / num_comp

    return cell_scores, cell_scores_positive, cell_scores_negative, cell_scores_exclusive, p_same_registered_pairs


def cluster_cells_matlab(cell_to_index_map: np.ndarray,
                         all_to_all_p_same,
                         all_to_all_indexes,
                         maximal_distance: float,
                         registration_threshold: float,
                         centroid_locations,
                         registration_approach: str = "Probabilistic",
                         transform_data: bool = False,
                         maximal_number_of_iterations: int = 10,
                         num_changes_thresh: int = 10,
                         decision_type: str = "Maximal similarity",
                         verbose: bool = True):
    """
    Faithful port of CellReg cluster_cells.m (split/switch/move/delete/merge).
    centroid_locations: list of arrays (n_cells,2) per session, in same units as maximal_distance.
    """
    cluster_distance_threshold = 1.7 * maximal_distance
    number_of_sessions = len(centroid_locations)

    cmap = np.array(cell_to_index_map, dtype=int, copy=True)

    def _cluster_centroids(cmap_local):
        ncl = cmap_local.shape[0]
        cc = np.zeros((ncl, 2), dtype=float)
        for i in range(ncl):
            cells = cmap_local[i, :]
            sess = np.where(cells > 0)[0]
            if sess.size == 0:
                cc[i, :] = 1e9
                continue
            pts = np.vstack([centroid_locations[s][cells[s] - 1, :] for s in sess])
            cc[i, :] = pts.mean(axis=0)
        return cc

    def _similarity(sess_a, cell_a_1idx, sess_b, cell_b_1idx):
        if cell_a_1idx <= 0 or cell_b_1idx <= 0:
            return None
        cand = all_to_all_indexes[sess_a][cell_a_1idx - 1][sess_b]
        ps = all_to_all_p_same[sess_a][cell_a_1idx - 1][sess_b]
        if cand is None or ps is None:
            return None
        cand = np.asarray(cand, dtype=int).ravel()
        ps = np.asarray(ps, dtype=float).ravel()
        hit = np.where(cand == cell_b_1idx)[0]
        if hit.size == 0:
            return None
        val = float(ps[hit[0]])
        if transform_data:
            return transform_distance_to_similarity(val, maximal_distance)
        return val

    changes_history = []

    if verbose:
        print("Clustering cells (MATLAB-faithful)...")

    for it in range(maximal_number_of_iterations):
        changes = 0

        num_clusters = cmap.shape[0]
        clusters_centroids = _cluster_centroids(cmap)

        # reassignment / split / switch
        for n in range(number_of_sessions):
            cluster_ind = np.where(cmap[:, n] > 0)[0]
            num_cells = cluster_ind.size

            for kk in range(num_cells):
                orig_cluster = int(cluster_ind[kk])
                this_cell = int(cmap[orig_cluster, n])
                if this_cell <= 0:
                    continue

                this_centroid = centroid_locations[n][this_cell - 1, :]
                dvec = np.sqrt(np.sum((clusters_centroids - this_centroid) ** 2, axis=1))
                clusters_to_check = np.where(dvec < cluster_distance_threshold)[0]
                if clusters_to_check.size == 0:
                    clusters_to_check = np.array([orig_cluster], dtype=int)

                num_candidates = clusters_to_check.size
                total_similarity = np.zeros(num_candidates, dtype=float)
                norm_factor = np.zeros(num_candidates, dtype=float)
                max_in_cluster = np.zeros(num_candidates, dtype=float)
                min_in_cluster = np.ones(num_candidates, dtype=float)

                for ci, cl in enumerate(clusters_to_check):
                    cells_in_cl = cmap[cl, :]
                    sess_in_cl = np.where(cells_in_cl > 0)[0]
                    sess_in_cl = sess_in_cl[sess_in_cl != n]

                    if sess_in_cl.size == 0:
                        min_in_cluster[ci] = 0.0
                        continue

                    for s2 in sess_in_cl:
                        in_cluster_cell = int(cmap[cl, s2])
                        sim = _similarity(n, this_cell, s2, in_cluster_cell)
                        if sim is None:
                            min_in_cluster[ci] = 0.0
                        else:
                            total_similarity[ci] += sim
                            norm_factor[ci] += 1
                            max_in_cluster[ci] = max(max_in_cluster[ci], sim)
                            min_in_cluster[ci] = min(min_in_cluster[ci], sim)

                if decision_type == "Minimal dissimilarity":
                    max_sim = float(np.max(min_in_cluster))
                    max_idx = int(np.argmax(min_in_cluster))
                elif decision_type == "Average similarity":
                    normed = np.divide(total_similarity, norm_factor, out=np.zeros_like(total_similarity), where=norm_factor > 0)
                    max_sim = float(np.max(normed))
                    max_idx = int(np.argmax(normed))
                else:  # "Maximal similarity"
                    max_sim = float(np.max(max_in_cluster))
                    max_idx = int(np.argmax(max_in_cluster))

                best_cluster = int(clusters_to_check[max_idx])
                best_cells = cmap[best_cluster, :]
                best_sess = np.where(best_cells > 0)[0]
                best_sess = best_sess[best_sess != n]

                if best_sess.size == 0:
                    continue

                original_cells = cmap[orig_cluster, :]
                num_in_original = int(np.sum(original_cells > 0))

                average_similarity = max_sim  # MATLAB for Maximal/Minimal paths

                # split
                if average_similarity < registration_threshold and num_in_original > 1:
                    cmap = np.vstack([cmap, np.zeros((1, number_of_sessions), dtype=int)])
                    cmap[-1, n] = this_cell
                    cmap[orig_cluster, n] = 0
                    clusters_centroids = np.vstack([clusters_centroids, this_centroid.reshape(1, 2)])
                    changes += 1
                    continue

                # move/switch
                if average_similarity >= registration_threshold:
                    if best_cluster != orig_cluster:
                        # does best_cluster already contain a cell from session n?
                        if cmap[best_cluster, n] > 0:
                            temp_cell = int(cmap[best_cluster, n])
                            temp_similarity = 0.0
                            # MATLAB sums similarities for the temp_cell
                            temp_cells = cmap[best_cluster, :]
                            temp_sess = np.where(temp_cells > 0)[0]
                            temp_sess = temp_sess[temp_sess != n]
                            for s2 in temp_sess:
                                in_cluster_cell = int(cmap[best_cluster, s2])
                                sim = _similarity(n, temp_cell, s2, in_cluster_cell)
                                if sim is not None:
                                    temp_similarity += sim

                            if max_sim > temp_similarity:
                                cmap = np.vstack([cmap, np.zeros((1, number_of_sessions), dtype=int)])
                                cmap[best_cluster, n] = this_cell
                                cmap[-1, n] = temp_cell
                                cmap[orig_cluster, n] = 0
                                clusters_centroids = np.vstack([clusters_centroids,
                                                                centroid_locations[n][temp_cell - 1, :].reshape(1, 2)])
                                changes += 1
                        else:
                            cmap[orig_cluster, n] = 0
                            cmap[best_cluster, n] = this_cell
                            changes += 1

        # delete empties
        before = cmap.shape[0]
        cmap = cmap[np.sum(cmap, axis=1) > 0, :]
        if cmap.shape[0] != before:
            changes += (before - cmap.shape[0])

        # merge step (MATLAB: Maximal similarity criterion by default)
        cmap_temp = cmap.copy()
        num_clusters = cmap_temp.shape[0]
        clusters_centroids = _cluster_centroids(cmap_temp)

        for i in range(num_clusters):
            this_cells = cmap_temp[i, :]
            this_sess = np.where(this_cells > 0)[0]
            if this_sess.size == 0:
                continue

            dvec = np.sqrt(np.sum((clusters_centroids - clusters_centroids[i, :]) ** 2, axis=1))
            candidates = np.where(dvec < cluster_distance_threshold)[0]
            candidates = candidates[candidates != i]

            for j in candidates:
                cand_cells = cmap_temp[j, :]
                cand_sess = np.where(cand_cells > 0)[0]
                if cand_sess.size == 0:
                    continue

                # only if no overlapping sessions
                if np.intersect1d(this_sess, cand_sess).size != 0:
                    continue

                # compute all_to_all_temp similarities
                sims = []
                for sa in this_sess:
                    ca = int(this_cells[sa])
                    for sb in cand_sess:
                        cb = int(cand_cells[sb])
                        sim = _similarity(sa, ca, sb, cb)
                        if sim is not None:
                            sims.append(sim)

                if len(sims) == 0:
                    continue

                if decision_type == "Maximal similarity":
                    if float(np.max(sims)) > registration_threshold:
                        cmap_temp[i, cand_sess] = cmap_temp[j, cand_sess]
                        cmap_temp[j, :] = 0
                        clusters_centroids[j, :] = 1e9
                        changes += 1

        cmap = cmap_temp
        before = cmap.shape[0]
        cmap = cmap[np.sum(cmap, axis=1) > 0, :]
        if cmap.shape[0] != before:
            changes += (before - cmap.shape[0])

        changes_history.append(changes)
        if verbose:
            print(f"  iter {it+1}: changes={changes}, clusters={cmap.shape[0]}")

        if it > 0 and changes <= num_changes_thresh:
            break

    clusters_centroids = _cluster_centroids(cmap)

    cluster_scores = {}
    if registration_approach == "Probabilistic":
        scores = compute_scores_matlab(cmap, all_to_all_indexes, all_to_all_p_same, number_of_sessions)
        cluster_scores = {
            "cell_scores": scores[0],
            "cell_scores_positive": scores[1],
            "cell_scores_negative": scores[2],
            "cell_scores_exclusive": scores[3],
            "p_same_registered_pairs": scores[4],
            "changes_history": changes_history,
        }

    return cmap, clusters_centroids, cluster_scores


def combine_p_same(p_same_a, p_same_b):
    """
    Combine two p_same structures by element-wise multiplication.

    Each p_same is a nested list:
        p_same[sess_i][cell_j][sess_k] = 1-D array of p_same values

    Returns a new structure with the same shape where each element is
    p_same_a * p_same_b.  Entries that exist in only one input are set
    to the value from that input (graceful fallback).
    """
    n_sessions = len(p_same_a)
    combined = []
    for si in range(n_sessions):
        sess_list = []
        for ci in range(len(p_same_a[si])):
            cell_dict = {}
            for sk in range(n_sessions):
                pa = p_same_a[si][ci].get(sk) if isinstance(p_same_a[si][ci], dict) else (
                    p_same_a[si][ci][sk] if sk < len(p_same_a[si][ci]) else None)
                pb = p_same_b[si][ci].get(sk) if isinstance(p_same_b[si][ci], dict) else (
                    p_same_b[si][ci][sk] if sk < len(p_same_b[si][ci]) else None)

                if pa is None and pb is None:
                    val = None
                elif pa is None:
                    val = pb
                elif pb is None:
                    val = pa
                else:
                    a_arr = np.asarray(pa, dtype=np.float64)
                    b_arr = np.asarray(pb, dtype=np.float64)
                    # Arrays must match in length;
                    # if not, take element-wise min length
                    n = min(len(a_arr), len(b_arr))
                    val = a_arr[:n] * b_arr[:n]

                cell_dict[sk] = val
            sess_list.append(cell_dict)
        combined.append(sess_list)
    return combined


def cluster_cells_consensus(cell_to_index_map,
                            p_same_centroid,
                            p_same_spatial,
                            all_to_all_indexes,
                            maximal_distance,
                            registration_threshold,
                            centroid_locations,
                            registration_approach='Probabilistic',
                            transform_data=False,
                            verbose=True):
    """
    Consensus clustering: run cluster_cells_matlab independently with
    centroid p_same and spatial p_same, then keep only cells that appear
    in BOTH cluster maps.

    Returns:
        consensus_map : np.ndarray, shape (n_consensus, n_sessions)
            Cell-to-index map containing only consensus clusters.
        map_centroid  : np.ndarray  – full centroid-only cluster map
        map_spatial   : np.ndarray  – full spatial-only cluster map
        n_centroid    : int – total clusters from centroid model
        n_spatial     : int – total clusters from spatial model
    """
    if verbose:
        print("  [Consensus] Running centroid-model clustering...")
    map_c, _, _ = cluster_cells_matlab(
        cell_to_index_map=cell_to_index_map,
        all_to_all_p_same=p_same_centroid,
        all_to_all_indexes=all_to_all_indexes,
        maximal_distance=maximal_distance,
        registration_threshold=registration_threshold,
        centroid_locations=centroid_locations,
        registration_approach=registration_approach,
        transform_data=transform_data,
        verbose=False,
    )

    if verbose:
        print("  [Consensus] Running spatial-model clustering...")
    map_s, _, _ = cluster_cells_matlab(
        cell_to_index_map=cell_to_index_map,
        all_to_all_p_same=p_same_spatial,
        all_to_all_indexes=all_to_all_indexes,
        maximal_distance=maximal_distance,
        registration_threshold=registration_threshold,
        centroid_locations=centroid_locations,
        registration_approach=registration_approach,
        transform_data=transform_data,
        verbose=False,
    )

    n_sessions = map_c.shape[1]

    # Build lookup: (session, cell_1based) -> cluster_row for each map
    def _build_lookup(cmap):
        lookup = {}
        for row in range(cmap.shape[0]):
            for s in range(cmap.shape[1]):
                idx = int(cmap[row, s])
                if idx > 0:
                    lookup[(s, idx)] = row
        return lookup

    lookup_c = _build_lookup(map_c)
    lookup_s = _build_lookup(map_s)

    # For each cluster in map_c, check if the SAME cells also form a
    # cluster in map_s (i.e., all present cells map to the same row in map_s)
    consensus_rows = []
    for row_c in range(map_c.shape[0]):
        cells_c = map_c[row_c, :]
        present = np.where(cells_c > 0)[0]
        if len(present) < 2:
            continue  # single-session cluster, skip

        # Find what rows these cells map to in map_s
        s_rows = set()
        all_found = True
        for s in present:
            key = (s, int(cells_c[s]))
            if key in lookup_s:
                s_rows.add(lookup_s[key])
            else:
                all_found = False
                break

        # Consensus: all cells in this centroid cluster must belong
        # to the SAME cluster in the spatial map
        if all_found and len(s_rows) == 1:
            consensus_rows.append(row_c)

    if len(consensus_rows) > 0:
        consensus_map = map_c[consensus_rows, :]
    else:
        consensus_map = np.zeros((0, n_sessions), dtype=int)

    if verbose:
        print(f"  [Consensus] Centroid clusters: {map_c.shape[0]}, "
              f"Spatial clusters: {map_s.shape[0]}, "
              f"Consensus: {consensus_map.shape[0]}")

    return consensus_map, map_c, map_s, map_c.shape[0], map_s.shape[0]


def cluster_cells(cell_to_index_map: np.ndarray,
                  all_to_all_p_same: List,
                  all_to_all_indexes: List,
                  normalized_maximal_distance: float,
                  threshold: float,
                  centroid_locations_corrected: List[np.ndarray],
                  registration_approach: str = 'Probabilistic',
                  transform_data: bool = False,
                  max_passes: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Practical Python port of MATLAB cluster_cells.m.

    This is a deterministic, MATLAB-faithful *approximation*:
    - For each existing cluster row and each present session-cell, greedily adds the
      highest-scoring candidate in other sessions if above threshold.
    - Enforces 1-cell-per-session exclusivity by keeping the cluster-row with the best
      mean pair score for collisions.

    Returns:
      cell_to_index_map (updated),
      registered_cells_centroids (n_clusters x 2),
      cluster_scores (n_clusters,)
    """
    cmap = np.array(cell_to_index_map, copy=True)
    n_clusters, n_sessions = cmap.shape

    # Helper: lookup score between (s,i) and (t,j)
    def pair_score(s, i, t, j):
        if i <= 0 or j <= 0:
            return 0.0
        cand = all_to_all_indexes[s][i - 1][t]
        ps = all_to_all_p_same[s][i - 1][t]
        if cand is None or ps is None:
            return 0.0
        cand = np.asarray(cand, dtype=int)
        ps = np.asarray(ps, dtype=np.float64)
        k = np.where(cand == j)[0]
        if k.size == 0:
            return 0.0
        val = float(ps[int(k[0])])
        if transform_data:
            # if data are distances, convert to similarity
            val = float(_transform_distance_to_similarity(val, normalized_maximal_distance))
        return val

    for _ in range(max_passes):
        # Expansion pass: for each cluster, try to fill missing sessions
        for r in range(n_clusters):
            present_sessions = np.where(cmap[r, :] > 0)[0]
            if present_sessions.size == 0:
                continue
            for s in present_sessions:
                i = int(cmap[r, s])
                for t in range(n_sessions):
                    if t == s:
                        continue
                    if cmap[r, t] > 0:
                        continue
                    cand = all_to_all_indexes[s][i - 1][t]
                    ps = all_to_all_p_same[s][i - 1][t]
                    if cand is None or ps is None:
                        continue
                    cand = np.asarray(cand, dtype=int)
                    ps = np.asarray(ps, dtype=np.float64)
                    if cand.size == 0:
                        continue
                    # choose best candidate
                    k = int(np.argmax(ps))
                    best = float(ps[k])
                    if best >= threshold:
                        cmap[r, t] = int(cand[k])

        # Exclusivity enforcement: within each session, one cell index per cluster
        for s in range(n_sessions):
            col = cmap[:, s]
            cells = col[col > 0]
            if cells.size == 0:
                continue
            # map cell -> rows
            unique_cells = np.unique(cells)
            for ci in unique_cells:
                rows = np.where(col == ci)[0]
                if rows.size <= 1:
                    continue
                # pick best row based on mean pair score of this (s,ci) to other present cells
                best_row = None
                best_score = -np.inf
                for r in rows:
                    # average score to all other sessions present in that row
                    others = np.where((cmap[r, :] > 0) & (np.arange(n_sessions) != s))[0]
                    if others.size == 0:
                        score = 0.0
                    else:
                        vals = [pair_score(s, int(ci), t, int(cmap[r, t])) for t in others]
                        score = float(np.mean(vals)) if len(vals) else 0.0
                    if score > best_score:
                        best_score = score
                        best_row = int(r)
                # zero-out ci in other rows
                for r in rows:
                    if int(r) != best_row:
                        cmap[int(r), s] = 0

    # Compute cluster centroids and cluster scores
    centroids = np.full((n_clusters, 2), np.nan, dtype=np.float64)
    cluster_scores = np.full((n_clusters,), np.nan, dtype=np.float64)

    for r in range(n_clusters):
        pts = []
        # score: mean pairwise score among present sessions
        pres = np.where(cmap[r, :] > 0)[0]
        scores = []
        for ii, s in enumerate(pres):
            i = int(cmap[r, s])
            # centroid point
            pts.append(centroid_locations_corrected[s][i - 1])
            for t in pres[ii + 1:]:
                j = int(cmap[r, t])
                scores.append(pair_score(s, i, t, j))
        if pts:
            pts = np.asarray(pts, dtype=np.float64)
            centroids[r, :] = np.nanmean(pts, axis=0)
        if scores:
            cluster_scores[r] = float(np.mean(scores))
        else:
            cluster_scores[r] = 0.0

    return cmap, centroids, cluster_scores




# ============================================================================ #
#                              MODULE EXPORTS                                  #
# ============================================================================ #



# ============================================================================ #
#                             CONVENIENCE API                                  #
# ============================================================================ #

def run_pipeline(folder_path: Union[str, Path],
                 cfg: Optional[CellRegConfig] = None,
                 *,
                 spatial_corr_floor: float = 0.5,
                 save_figures: bool = True,
                 figures_visibility: str = 'off',
                 export_csv: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """One-call entrypoint: run the full CellRegPy pipeline on a mouse folder.

    This enforces the Stage-6 dual-model approach:
        1) centroid-distance probabilistic model drives clustering
        2) spatial correlation is used ONLY as a post-hoc veto (floor cutoff)

    Args:
        folder_path: Mouse folder containing session subfolders.
        cfg: Optional CellRegConfig. If None, defaults are used.
        spatial_corr_floor: Spatial correlation veto threshold (default 0.5).
        save_figures: Save per-FOV figures into 1_CellReg/FOV*/Figures (default True).
        figures_visibility: 'on' to display; 'off' to save+close (default 'off').
        export_csv: Save mouse_table.csv and mouse_table_wide.csv into 1_CellReg (default True).

    Returns:
        (mouse_table, mouse_data)
    """
    mouse_folder = Path(folder_path)
    cfg = cfg or CellRegConfig()

    # Enforce dual-model final registration
    cfg.model_type = 'Centroid distance'
    cfg.dual_model = True
    cfg.apply_spatial_floor_filter = True
    cfg.spatial_corr_floor = float(spatial_corr_floor)

    # Figures
    cfg.save_figures = bool(save_figures)
    cfg.figures_visibility = str(figures_visibility)
    cfg.close_figures = True

    cellreg = CellRegPy(cfg)
    mouse_table, mouse_data = cellreg.run([mouse_folder])

    if export_csv:
        try:
            out_dir = mouse_folder / '1_CellReg'
            out_dir.mkdir(exist_ok=True)
            mouse_table.to_csv(out_dir / 'mouse_table.csv', index=False)
            wide = (mouse_table
                    .pivot_table(index='cellRegID', columns='Session', values='suite2pID',
                                 aggfunc='first', fill_value=0)
                    .sort_index())
            wide.to_csv(out_dir / 'mouse_table_wide.csv')
        except Exception:
            pass

    return mouse_table, mouse_data

__all__ = [
    # Main classes
    'CellRegConfig',
    'CellRegPy',
    'run_pipeline',
    'MeanImageAligner',
    # Data loading
    'load_cellreg_mat',
    'load_fall_mat',
    'get_mean_image',
    'get_spatial_footprints',
    'get_iscell',
    'list_session_folders',
    'get_cellreg_files',
    # Spatial footprint processing
    'normalize_footprints',
    'adjust_fov_size',
    'compute_footprint_projections',
    'compute_centroids',
    'compute_centroid_projections',
    'make_alignment_image_from_footprints',
    'gaussfit',
    # Probabilistic modeling & registration
    'estimate_num_bins',
    'compute_spatial_correlation',
    'compute_data_distribution',
    'initial_registration_spatial_corr',
    'compute_p_same',
]
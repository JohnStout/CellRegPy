"""
CellRegPy – Cross-session cell registration for calcium imaging.

A Python port of the MATLAB CellReg/batchRunCellReg pipeline with key
improvements including mean-image alignment and probabilistic cell matching.
"""

__version__ = "0.1.0"

from .cellregpy import (
    # Configuration
    CellRegConfig,
    # Data loading
    load_cellreg_mat,
    load_fall_mat,
    get_mean_image,
    get_spatial_footprints,
    get_iscell,
    list_session_folders,
    get_cellreg_files,
    # Helpers
    normalize_footprints,
    adjust_fov_size,
    compute_footprint_projections,
    compute_centroids,
    suite2pToCellReg,
    compute_centroid_projections,
    # Probabilistic modelling
    estimate_num_bins,
    compute_data_distribution,
    compute_centroid_distances_model_custom,
    compute_spatial_correlations_model,
    compute_p_same,
    choose_best_model,
    # Registration
    initial_registration_centroid_distances_custom,
    initial_registration_spatial_corr,
    cluster_cells_matlab,
    estimate_registration_accuracy,
    # Alignment
    MeanImageAligner,
)

__all__ = [
    "__version__",
    "CellRegConfig",
    "load_cellreg_mat",
    "load_fall_mat",
    "get_mean_image",
    "get_spatial_footprints",
    "get_iscell",
    "list_session_folders",
    "get_cellreg_files",
    "normalize_footprints",
    "adjust_fov_size",
    "compute_footprint_projections",
    "compute_centroids",
    "compute_centroid_projections",
    "suite2pToCellReg",
    "estimate_num_bins",
    "compute_data_distribution",
    "compute_centroid_distances_model_custom",
    "compute_spatial_correlations_model",
    "compute_p_same",
    "choose_best_model",
    "initial_registration_centroid_distances_custom",
    "initial_registration_spatial_corr",
    "cluster_cells_matlab",
    "estimate_registration_accuracy",
    "MeanImageAligner",
]

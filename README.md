# CellRegPy

**Cross-session cell registration for calcium imaging in Python. Works with Suite2P!**

A Python port of the MATLAB [CellReg](https://github.com/zivlab/CellReg) with key improvements meant to fit with Suite2P results:

- **Mean image alignment** — multi-transform search with high-pass filtering (innovation over standard CellReg)
- **Probabilistic cell matching** — based on spatial footprints and centroid distances
- **Automatic session detection** — finds suite2p `plane0` folders from a mouse directory - Currently only supports single plane imaging
- **MATLAB-matching figures** — reproduces all CellReg diagnostic figures
- **Combined modeling approach** - Returns aligned cells using multiple different methods for detecting cells, including a new combined centroid-distance + spatial correlation filtering method!

## Installation

### Option 1: Conda environment (recommended)

This creates a self-contained `cellregpy` environment with all dependencies:

```bash
git clone https://github.com/JohnStout/CellRegPy.git
cd CellRegPy
conda env create -f environment.yml
conda activate cellregpy
```

### Option 2: pip install from source

```bash
git clone https://github.com/JohnStout/CellRegPy.git
cd CellRegPy
pip install -e .
```

### Option 3: pip install from PyPI

```bash
pip install cellregpy
```

## Quick Start

```python
from cellregpy import CellRegConfig, list_session_folders, get_spatial_footprints, MeanImageAligner

# Point to your mouse data folder
mouse_folder = r"path/to/mouse/data"

# Find all sessions
sessions = list_session_folders(mouse_folder)
print(f"Found {len(sessions)} sessions")

# Load spatial footprints
footprints = [get_spatial_footprints(s / "CellReg.mat") for s in sessions]

# Align sessions using mean images
aligner = MeanImageAligner(CellRegConfig(microns_per_pixel=2.0))
```

See [`notebooks/demo_validate_alignment.ipynb`](notebooks/demo_validate_alignment.ipynb) for a full end-to-end example.

## Package Structure

```
CellRegPy/
├── cellregpy/
│   ├── __init__.py              # Public API
│   ├── cellregpy.py             # Core registration engine
│   └── plotting.py              # MATLAB-matching figure generation
├── notebooks/
│   └── demo_validate_alignment.ipynb   # End-to-end demo notebook
├── pyproject.toml
├── LICENSE
└── README.md
```

## Dependencies

- NumPy ≥ 1.22
- SciPy ≥ 1.8
- scikit-image ≥ 0.19
- Matplotlib ≥ 3.5
- pandas ≥ 1.4

## Citation

If you use CellRegPy in your research, please cite the original CellReg research article:

> Sheintuch, L., Rubin, A., Brande-Eilat, N., Geva, N., Sadeh, N., Pinchasof, O., Ziv, Y. (2017). Tracking the Same Neurons across Multiple Days in Ca2+ Imaging Data. Cell Reports, 21(4), pp. 1102–1115. doi: 10.1016/j.celrep.2017.10.013.

## License

MIT License — see [LICENSE](LICENSE) for details.

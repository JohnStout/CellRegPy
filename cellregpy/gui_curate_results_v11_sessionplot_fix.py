
"""
CellRegPy post-processing curation GUI

Goal:
- Load an existing CellRegPy output mouse_table (CSV/PKL) from a mouse folder.
- Visualize mean images + ROI outlines for each session (suite2p plane0).
- Click/select a row to view the same CellRegID across sessions.
- Optional: compute an alternative deterministic registration using IoU + Hungarian
  as a *post-processing* step (on-demand).
- Manual curation: link/unlink selected rows into curated clusters.
- Save a curated mouse_table (CSV + PKL) without overwriting the original.

This is intentionally a "validation + curation" tool, not a replacement for run_pipeline.
"""

from __future__ import annotations

import threading
import queue
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")  # GUI backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    linear_sum_assignment = None
    warnings.warn(f"scipy not available; IoU+Hungarian disabled: {e}")


# ---------------------------
# Helpers
# ---------------------------

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _truncate(s: str, n: int = 36) -> str:
    s = str(s)
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(x):
            return default
        v = int(float(x))
        return v
    except Exception:
        return default


def _norm_img(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-normalize for display."""
    a = np.asarray(img, dtype=float)
    if a.size == 0:
        return a
    lo = np.nanpercentile(a, p_low)
    hi = np.nanpercentile(a, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return a
    a = (a - lo) / (hi - lo)
    return np.clip(a, 0, 1)


def _load_npy(path: Path) -> Any:
    return np.load(path, allow_pickle=True)


def _suite2p_plane0_from_session(session_root: Path) -> Path:
    """
    Given a 'SessionPath' from mouse_table (session root), locate suite2p/plane0.
    """
    # Most of your data uses session_root/suite2p/plane0
    cand = session_root / "suite2p" / "plane0"
    if cand.exists():
        return cand
    # Fallback: maybe session_root already points to plane0
    if (session_root / "ops.npy").exists():
        return session_root
    # Try one more common pattern
    cand2 = session_root / "plane0"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Could not locate suite2p plane0 for session: {session_root}")


def _roi_polygon_from_stat(stat_row: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Build a polygon outline from suite2p stat entry.
    Uses stat['xpix'], stat['ypix'] to create a hull-ish contour by sorting on angle
    around centroid. It's not perfect but good for interactive selection/overlay.
    """
    if stat_row is None:
        return None
    xpix = stat_row.get("xpix", None)
    ypix = stat_row.get("ypix", None)
    if xpix is None or ypix is None:
        return None
    x = np.asarray(xpix, dtype=float)
    y = np.asarray(ypix, dtype=float)
    if x.size < 6:
        return None
    cx = float(np.nanmean(x))
    cy = float(np.nanmean(y))
    ang = np.arctan2(y - cy, x - cx)
    order = np.argsort(ang)
    poly = np.column_stack([x[order], y[order]])
    return poly


def _roi_centroid_from_stat(stat_row: Dict[str, Any]) -> Tuple[float, float]:
    """
    Suite2p stat typically has 'med' (y,x) or can infer centroid from xpix/ypix.
    Returns (x, y).
    """
    if stat_row is None:
        return (np.nan, np.nan)
    if "med" in stat_row:
        med = np.asarray(stat_row["med"], dtype=float).ravel()
        if med.size >= 2:
            # suite2p uses med = [y, x]
            return (float(med[1]), float(med[0]))
    xpix = stat_row.get("xpix", None)
    ypix = stat_row.get("ypix", None)
    if xpix is not None and ypix is not None:
        return (float(np.nanmean(xpix)), float(np.nanmean(ypix)))
    return (np.nan, np.nan)


def _roi_mask_indices_from_stat(
    stat_row: Dict[str, Any],
    shape_hw: Tuple[int, int],
    mask_threshold: float = 0.15,
) -> np.ndarray:
    """
    Convert suite2p stat entry into a 1D array of linear pixel indices for a thresholded mask.
    Uses lam weights when present; else uses all pixels in xpix/ypix.
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])
    xpix = stat_row.get("xpix", None)
    ypix = stat_row.get("ypix", None)
    if xpix is None or ypix is None:
        return np.zeros((0,), dtype=np.int32)

    x = np.asarray(xpix, dtype=np.int32)
    y = np.asarray(ypix, dtype=np.int32)

    if "lam" in stat_row and stat_row["lam"] is not None:
        lam = np.asarray(stat_row["lam"], dtype=float).ravel()
        if lam.size == x.size:
            m = float(np.nanmax(lam)) if lam.size else 0.0
            if np.isfinite(m) and m > 0:
                keep = lam >= (mask_threshold * m)
                x = x[keep]
                y = y[keep]

    # clip
    keep2 = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[keep2]
    y = y[keep2]
    if x.size == 0:
        return np.zeros((0,), dtype=np.int32)
    idx = (y.astype(np.int64) * W + x.astype(np.int64)).astype(np.int64)
    idx = np.unique(idx)
    return idx.astype(np.int32)


def _roi_footprint_image_from_stat(stat_row: Dict[str, Any], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Build a dense footprint image for display from suite2p stat.
    Uses lam weights when present; otherwise uses a binary mask.
    """
    if stat_row is None:
        return None
    H, W = int(shape_hw[0]), int(shape_hw[1])
    xpix = stat_row.get("xpix", None)
    ypix = stat_row.get("ypix", None)
    if xpix is None or ypix is None:
        return None
    x = np.asarray(xpix, dtype=np.int32)
    y = np.asarray(ypix, dtype=np.int32)
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        return None
    fp = np.zeros((H, W), dtype=float)
    if "lam" in stat_row and stat_row["lam"] is not None:
        lam = np.asarray(stat_row["lam"], dtype=float).ravel()
        if lam.size == keep.size:
            lam = lam[keep]
        elif lam.size == x.size:
            pass
        else:
            lam = np.ones_like(x, dtype=float)
        fp[y, x] = lam
    else:
        fp[y, x] = 1.0
    return fp

def _iou_from_sorted_indices(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two sorted unique 1D int arrays of pixel indices."""
    if a.size == 0 or b.size == 0:
        return 0.0
    # intersection via two-pointer (fast for small arrays)
    i = j = 0
    inter = 0
    na = int(a.size)
    nb = int(b.size)
    while i < na and j < nb:
        va = int(a[i])
        vb = int(b[j])
        if va == vb:
            inter += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    union = na + nb - inter
    return float(inter) / float(union) if union > 0 else 0.0


# ---------------------------
# Session cache
# ---------------------------

@dataclass
class SessionData:
    session_name: str
    session_root: Path
    plane0: Path
    ops: Dict[str, Any]
    stat: List[Dict[str, Any]]
    iscell: np.ndarray
    bg: np.ndarray  # mean image (motion corrected)
    bg_desc: str
    shape_hw: Tuple[int, int]


class SessionCache:
    """
    Lazy-load suite2p session data. Keeps in-memory cache to avoid repeated disk hits.
    """

    def __init__(self, logger=None):
        self._cache: Dict[str, SessionData] = {}
        self._logger = logger

    def set_logger(self, logger):
        self._logger = logger

    def log(self, msg: str):
        if self._logger:
            self._logger(msg)

    def get(self, session_name: str, session_root: Path) -> SessionData:
        key = str(session_root.resolve())
        if key in self._cache:
            return self._cache[key]

        t0 = time.time()
        self.log(f"[cache] loading {session_name} | {session_root}")
        plane0 = _suite2p_plane0_from_session(session_root)

        # Load ops/stat/iscell
        ops = _load_npy(plane0 / "ops.npy").item()
        stat = _load_npy(plane0 / "stat.npy").tolist()
        iscell = _load_npy(plane0 / "iscell.npy")
        if iscell.ndim == 2:
            iscell_bool = iscell[:, 0].astype(bool)
        else:
            iscell_bool = iscell.astype(bool)

        # Prefer motion-corrected mean image (meanImg) not enhanced meanImgE
        bg = None
        bg_desc = "meanImg"
        if "meanImg" in ops and ops["meanImg"] is not None:
            bg = np.asarray(ops["meanImg"], dtype=float)
        elif "meanImgE" in ops and ops["meanImgE"] is not None:
            bg = np.asarray(ops["meanImgE"], dtype=float)
            bg_desc = "meanImgE"
        else:
            # fallback to zeros
            Ly = int(ops.get("Ly", 0) or 0)
            Lx = int(ops.get("Lx", 0) or 0)
            bg = np.zeros((Ly, Lx), dtype=float)
            bg_desc = "zeros"

        shape_hw = (int(bg.shape[0]), int(bg.shape[1]))
        sd = SessionData(
            session_name=str(session_name),
            session_root=Path(session_root),
            plane0=plane0,
            ops=ops,
            stat=stat,
            iscell=iscell_bool,
            bg=bg,
            bg_desc=bg_desc,
            shape_hw=shape_hw,
        )
        self._cache[key] = sd
        self.log(f"[cache] loaded {session_name} in {time.time()-t0:.3f}s | bg={bg_desc} | shape={shape_hw}")
        return sd


# ---------------------------
# Main GUI
# ---------------------------

class CellRegCurator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CellRegPy Curator (post-processing)")
        self.geometry("1600x900")

        # Data
        self.mouse_folder: Optional[Path] = None
        self.table_path: Optional[Path] = None
        self.t: Optional[pd.DataFrame] = None
        self.current_session: Optional[str] = None
        self.current_fov: Optional[int] = None

        # Columns
        self.COL_ORIG = "cellRegID"
        self.COL_IOU = "cellRegID_iou"
        self.COL_CUR = "cellRegID_curated"

        # Session cache
        self.cache = SessionCache(logger=self._log)

        # UI state / guards
        self._suppress_table_event = False
        self._in_select = False  # re-entrancy guard
        self._ignore_table_select = 0  # consume next TreeviewSelect events triggered programmatically

        # popup window (embedded canvas, reused)
        self._popup_win = None
        self._popup_fig = None
        self._popup_canvas = None
        self._selected_row: Optional[int] = None
        self._selected_cellid: Optional[int] = None
        # Popup interactive selection (for manual curation)
        self._popup_axes_to_session: Dict[Any, str] = {}
        self._popup_candidate: Optional[Dict[str, Any]] = None
        self._popup_highlight: Dict[str, Any] = {}  # session -> artist
        self._popup_last_pick: Optional[Dict[str, Any]] = None
  # {'session': str, 'suite2pID': int, 'row_idx': int}
        self._popup_base_cid: Optional[int] = None
        self._popup_pick_tol_px: float = 20.0
        self._popup_mpl_cid: Optional[int] = None
        self._popup_status_var = tk.StringVar(value='Click an ROI in another session panel to select a candidate.')

        # Background worker
        self._worker_q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.after(50, self._poll_worker_queue)

        self._build_menu()
        self._build_layout()

        self._log("ready. File → Open Mouse Folder/Table")

    # ---- UI scaffold ----

    def _build_menu(self):
        m = tk.Menu(self)
        self.config(menu=m)

        file_m = tk.Menu(m, tearoff=0)
        m.add_cascade(label="File", menu=file_m)
        file_m.add_command(label="Open Mouse Folder (auto-detect mouse_table)", command=self._open_mouse_folder)
        file_m.add_command(label="Open mouse_table (CSV/PKL)", command=self._open_mouse_table)
        file_m.add_separator()
        file_m.add_command(label="Save curated mouse_table", command=self._save_curated)
        file_m.add_separator()
        file_m.add_command(label="Quit", command=self.destroy)

        tools_m = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Tools", menu=tools_m)
        tools_m.add_command(label="Compute IoU+Hungarian (this FOV)", command=self._compute_iou_async)
        tools_m.add_command(label="Use IoU as curated", command=self._apply_iou_as_curated)
        tools_m.add_command(label="Reset curated = original", command=self._reset_curated)

        curate_m = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Curate", menu=curate_m)
        curate_m.add_command(label="Link selected rows (curated)", command=self._link_selected_rows)
        curate_m.add_command(label="Unlink selected rows (curated)", command=self._unlink_selected_rows)

        help_m = tk.Menu(m, tearoff=0)
        m.add_cascade(label="Help", menu=help_m)
        help_m.add_command(label="Workflow", command=self._show_help)

    def _build_layout(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.lbl_mouse = ttk.Label(top, text="Mouse: (no table loaded)")
        self.lbl_mouse.pack(side=tk.LEFT)

        ttk.Label(top, text="   Session:").pack(side=tk.LEFT)
        self.session_var = tk.StringVar(value="")
        self.session_dd = ttk.Combobox(top, textvariable=self.session_var, width=42, state="readonly")
        self.session_dd.pack(side=tk.LEFT, padx=(4, 8))
        self.session_dd.bind("<<ComboboxSelected>>", lambda e: self._on_session_change())

        self.cells_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Cells only (iscell)", variable=self.cells_only_var, command=self._render).pack(side=tk.LEFT, padx=6)

        self.labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Labels", variable=self.labels_var, command=self._render).pack(side=tk.LEFT, padx=6)

        self.popups_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Popups", variable=self.popups_var).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Label mode:").pack(side=tk.LEFT)
        self.label_mode_var = tk.StringVar(value="both")
        self.label_mode_dd = ttk.Combobox(top, textvariable=self.label_mode_var, width=10, state="readonly",
                                          values=["cellRegID", "suite2pID", "both", "curated"])
        self.label_mode_dd.pack(side=tk.LEFT, padx=6)
        self.label_mode_dd.bind("<<ComboboxSelected>>", lambda e: self._render())

        # Middle layout
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Left: image
        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.fig, self.ax = plt.subplots(figsize=(6.2, 6.2))
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click_image)

        # Right: table + log
        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Table controls
        tctrl = ttk.Frame(right)
        tctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(tctrl, text="Table view:").pack(side=tk.LEFT)
        self.table_view_var = tk.StringVar(value="Current session")
        self.table_view_dd = ttk.Combobox(tctrl, textvariable=self.table_view_var, width=18, state="readonly",
                                          values=["Current session", "All sessions", "Selected cellRegID"])
        self.table_view_dd.pack(side=tk.LEFT, padx=6)
        self.table_view_dd.bind("<<ComboboxSelected>>", lambda e: self._populate_table())

        # Buttons
        ttk.Button(tctrl, text="Compute IoU (this FOV)", command=self._compute_iou_async).pack(side=tk.LEFT, padx=6)
        ttk.Button(tctrl, text="Use IoU→curated", command=self._apply_iou_as_curated).pack(side=tk.LEFT, padx=6)
        ttk.Button(tctrl, text="Reset curated", command=self._reset_curated).pack(side=tk.LEFT, padx=6)

        # Table
        self.tree = ttk.Treeview(right, columns=("Session", "suite2pID", "cellRegID", "curated", "fovID", "UnixTime"), show="headings", height=24)
        for col, w in [("Session", 210), ("suite2pID", 80), ("cellRegID", 80), ("curated", 80), ("fovID", 50), ("UnixTime", 90)]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="w")
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_table_select)

        # Log
        log_frame = ttk.LabelFrame(right, text="Debug / status")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        self.log_txt = tk.Text(log_frame, height=8, wrap="word")
        self.log_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_txt.configure(yscrollcommand=sb.set)

        # Status bar
        self.status = ttk.Label(self, text="", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _show_help(self):
        message = (
            "Workflow:\n"
            "1) File → Open Mouse Folder (or mouse_table)\n"
            "2) Pick a Session.\n"
            "3) Click an ROI (near its centroid) OR click a row in the table.\n"
            "   - Table view 'Selected cellRegID' shows that CellRegID across sessions.\n"
            "4) If the CellReg result looks wrong for this FOV, Tools → Compute IoU+Hungarian.\n"
            "   - Then 'Use IoU→curated' to adopt it.\n"
            "5) Manual curation:\n"
            "   - Multi-select rows in the table (Ctrl/Shift) and Curate → Link/Unlink.\n"
            "6) File → Save curated mouse_table.\n"
        )
        messagebox.showinfo("Help", message)

    # ---- Logging ----

    def _log(self, msg: str):
        try:
            self.log_txt.insert("end", f"[{_ts()}] {msg}\n")
            self.log_txt.see("end")
        except Exception:
            pass
        try:
            self.status.configure(text=msg)
        except Exception:
            pass

    # ---- Load data ----

    def _open_mouse_folder(self):
        folder = filedialog.askdirectory(title="Select mouse folder (contains sessions + 1_CellReg)")
        if not folder:
            return
        self.mouse_folder = Path(folder)
        self.lbl_mouse.configure(text=f"Mouse: {_truncate(self.mouse_folder.name, 60)}")
        # Try to find a mouse_table
        candidates = []
        for p in [
            self.mouse_folder / "mouse_table.pkl",
            self.mouse_folder / "mouse_table.csv",
            self.mouse_folder / "1_CellReg" / "mouse_table.pkl",
            self.mouse_folder / "1_CellReg" / "mouse_table.csv",
            self.mouse_folder / "1_CellReg" / "mouse_table_curated.pkl",
            self.mouse_folder / "1_CellReg" / "mouse_table_curated.csv",
        ]:
            if p.exists():
                candidates.append(p)
        if not candidates:
            messagebox.showwarning("Not found", "Could not find mouse_table.{pkl,csv} in this folder.\nUse File → Open mouse_table.")
            return
        # Prefer non-curated base table
        pref = None
        for p in candidates:
            if p.name == "mouse_table.pkl":
                pref = p
                break
        if pref is None:
            pref = candidates[0]
        self._load_mouse_table(pref)

    def _open_mouse_table(self):
        path = filedialog.askopenfilename(
            title="Select mouse_table (CSV/PKL)",
            filetypes=[("mouse_table", "*.pkl *.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_mouse_table(Path(path))

    def _load_mouse_table(self, path: Path):
        self.table_path = path
        self._log(f"loading mouse_table: {path}")
        if path.suffix.lower() == ".pkl":
            t = pd.read_pickle(path)
        else:
            t = pd.read_csv(path)

        # If mouse_folder wasn't set (opened table directly), infer it from the table path.
        if self.mouse_folder is None:
            try:
                if path.parent.name == '1_CellReg':
                    self.mouse_folder = path.parent.parent
                else:
                    self.mouse_folder = path.parent
                self.lbl_mouse.configure(text=f"Mouse: {_truncate(self.mouse_folder.name, 60)}")
            except Exception:
                pass

        # Normalize expected cols
        expected = {"Session", "suite2pID", "cellRegID", "fovID", "SessionPath"}
        missing = expected - set(t.columns)
        if missing:
            raise ValueError(f"mouse_table missing columns: {missing}")

        # Add curated/ioU columns if missing
        if self.COL_CUR not in t.columns:
            t[self.COL_CUR] = t[self.COL_ORIG]
        if self.COL_IOU not in t.columns:
            t[self.COL_IOU] = np.nan

        # Keep ints where possible
        t["suite2pID"] = pd.to_numeric(t["suite2pID"], errors="coerce")
        t["cellRegID"] = pd.to_numeric(t["cellRegID"], errors="coerce")
        t[self.COL_CUR] = pd.to_numeric(t[self.COL_CUR], errors="coerce")
        t["fovID"] = pd.to_numeric(t["fovID"], errors="coerce")

        # Robust SessionPath: fill missing/invalid paths using mouse_folder/session.
        try:
            if self.mouse_folder is not None:
                mf = Path(self.mouse_folder)
            else:
                mf = None

            # per-session best path
            sess_to_path = {}
            for sess, g in t.groupby("Session"):
                pick = None
                # prefer any existing valid path
                for v in g.get("SessionPath", pd.Series([], dtype=object)).dropna().unique().tolist():
                    sv = str(v).strip()
                    if not sv or sv.lower() in ("none", "nan"):
                        continue
                    pth = Path(sv)
                    if pth.exists():
                        pick = pth
                        break
                # fallback to mouse_folder/session
                if pick is None and mf is not None:
                    cand = mf / str(sess)
                    if cand.exists():
                        pick = cand
                sess_to_path[str(sess)] = str(pick) if pick is not None else None

            def _fill_sp(row):
                sv = str(row.get("Session", "")).strip()
                cur = row.get("SessionPath", None)
                scur = "" if cur is None or (isinstance(cur, float) and np.isnan(cur)) else str(cur).strip()
                if scur and scur.lower() not in ("none", "nan"):
                    pth = Path(scur)
                    if pth.exists():
                        return str(pth)
                # use sess map
                mapped = sess_to_path.get(sv, None)
                if mapped:
                    return mapped
                if mf is not None:
                    return str(mf / sv)
                return scur if scur else None

            t["SessionPath"] = t.apply(_fill_sp, axis=1)
        except Exception as e:
            self._log(f"WARNING: could not normalize SessionPath: {e}")

        self.t = t.reset_index(drop=True)
        self._log(f"mouse_table loaded: rows={len(self.t)} sessions={self.t['Session'].nunique()} fovs={self.t['fovID'].nunique()}")
        self._refresh_session_dropdown()

    def _refresh_session_dropdown(self):
        assert self.t is not None
        sessions = sorted(self.t["Session"].astype(str).unique().tolist())
        self.session_dd["values"] = sessions
        if sessions:
            self.session_var.set(sessions[0])
            self.current_session = sessions[0]
        # Choose default FOV
        fovs = self.t["fovID"].dropna().unique()
        if fovs.size:
            self.current_fov = int(np.nanmin(fovs))
        else:
            self.current_fov = None
        self._populate_table()
        self._render()

    def _resolve_session_root(self, session: str) -> Optional[Path]:
        """Resolve a session folder robustly (works even if SessionPath is missing/bad)."""
        if self.t is None:
            return None
        session = str(session)
        # 1) Any valid SessionPath from table
        try:
            sub = self.t[self.t["Session"].astype(str) == session]
            if not sub.empty and "SessionPath" in sub.columns:
                for v in sub["SessionPath"].dropna().unique().tolist():
                    sv = str(v).strip()
                    if not sv or sv.lower() in ("none", "nan"):
                        continue
                    p = Path(sv)
                    if p.exists():
                        return p
        except Exception:
            pass
        # 2) mouse_folder/session
        try:
            if self.mouse_folder is not None:
                cand = Path(self.mouse_folder) / session
                if cand.exists():
                    return cand
        except Exception:
            pass
        return None

    # ---- Rendering ----

    def _on_session_change(self):
        self.current_session = self.session_var.get()
        self._selected_row = None
        self._selected_cellid = None
        self._populate_table()
        self._render()

    def _get_rows_for_view(self) -> np.ndarray:
        assert self.t is not None
        view = self.table_view_var.get()
        session = self.current_session
        fov = self.current_fov

        tt = self.t
        if fov is not None:
            tt = tt[tt["fovID"].fillna(-1).astype(int) == int(fov)]

        if view == "All sessions":
            rows = tt.index.to_numpy()
            return rows

        if view == "Selected cellRegID" and self._selected_cellid is not None:
            cid = int(self._selected_cellid)
            rows = tt.index[tt[self.COL_CUR].fillna(-1).astype(int) == cid].to_numpy()
            return rows

        # Current session
        if session is None:
            return tt.index.to_numpy()
        rows = tt.index[tt["Session"].astype(str) == str(session)].to_numpy()
        return rows

    def _populate_table(self):
        if self.t is None:
            return
        rows = self._get_rows_for_view()
        # Clear
        self._suppress_table_event = True
        try:
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Insert
            for r in rows:
                sess = str(self.t.loc[r, "Session"])
                sid = _safe_int(self.t.loc[r, "suite2pID"], None)
                cid = _safe_int(self.t.loc[r, "cellRegID"], None)
                cur = _safe_int(self.t.loc[r, self.COL_CUR], None)
                fov = _safe_int(self.t.loc[r, "fovID"], None)
                ut = self.t.loc[r, "UnixTime"] if "UnixTime" in self.t.columns else None
                self.tree.insert("", "end", iid=str(int(r)), values=(sess, sid, cid, cur, fov, ut))
        finally:
            self._suppress_table_event = False

        self._log(f"table populated: {len(rows)} row(s) | view={self.table_view_var.get()}")

    def _render(self):
        if self.t is None or self.current_session is None:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, "File → Open Mouse Folder/Table", ha="center", va="center")
            self.canvas.draw_idle()
            return

        session = str(self.current_session)
        session_root = self._resolve_session_root(session)
        if session_root is None:
            self.ax.clear()
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, f"Could not resolve session path for: {session}", ha="center", va="center")
            self.canvas.draw_idle()
            return
        sd = self.cache.get(session, session_root)

        self.ax.clear()
        self.ax.set_axis_off()

        img = _norm_img(sd.bg, 1, 99.5)
        self.ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
        self.ax.set_title(f"{session} | bg={sd.bg_desc}", fontsize=10)

        # Overlay ROIs (from rows for this session and current fov)
        fov = self.current_fov
        show = self.t.copy()
        if fov is not None:
            show = show[show["fovID"].fillna(-1).astype(int) == int(fov)]
        show = show[show["Session"].astype(str) == session]

        if self.cells_only_var.get():
            # If table has iscell col, use it. Else use suite2p iscell for that ROI.
            if "iscell" in show.columns:
                show = show[show["iscell"].fillna(0).astype(int) == 1]
            else:
                # map suite2pID to sd.iscell (suite2pID is 1-based)
                ids = pd.to_numeric(show["suite2pID"], errors="coerce").fillna(-1).astype(int).to_numpy()
                keep = []
                for k in ids:
                    if k <= 0 or k - 1 >= sd.iscell.size:
                        keep.append(False)
                    else:
                        keep.append(bool(sd.iscell[k - 1]))
                show = show.iloc[np.where(np.asarray(keep))[0]]

        label_mode = self.label_mode_var.get()
        draw_labels = bool(self.labels_var.get())

        # Draw outlines
        for _, row in show.iterrows():
            roi_1b = _safe_int(row["suite2pID"], None)
            if roi_1b is None:
                continue
            rid0 = roi_1b - 1
            if rid0 < 0 or rid0 >= len(sd.stat):
                continue
            poly = _roi_polygon_from_stat(sd.stat[rid0])
            if poly is None:
                continue
            self.ax.plot(poly[:, 0], poly[:, 1], color="deepskyblue", linewidth=0.9, alpha=0.9)

            if draw_labels:
                cx, cy = _roi_centroid_from_stat(sd.stat[rid0])
                txt = ""
                if label_mode == "suite2pID":
                    txt = f"{roi_1b}"
                elif label_mode == "cellRegID":
                    txt = str(_safe_int(row["cellRegID"], ""))
                elif label_mode == "curated":
                    txt = str(_safe_int(row[self.COL_CUR], ""))
                else:  # both
                    txt = f"{_safe_int(row[self.COL_CUR], '')}|{roi_1b}"
                if txt:
                    self.ax.text(cx, cy, txt, color="gold", fontsize=9, ha="center", va="center")

        self.canvas.draw_idle()

    # ---- Interaction ----

    def _on_table_select(self, event=None):
        # Prevent selection recursion / storms
        if self._ignore_table_select > 0:
            self._ignore_table_select -= 1
            return
        if self._suppress_table_event or self._in_select:
            return
        sel = self.tree.selection()
        if not sel:
            return
        try:
            row_idx = int(sel[0])
        except Exception:
            return
        if self._selected_row is not None and int(self._selected_row) == int(row_idx):
            return
        self._select_row(row_idx, from_table=True)

    def _on_click_image(self, event):
        if self.t is None or self.current_session is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        session = str(self.current_session)
        fov = self.current_fov
        # Find rows for this session+fov
        show = self.t.copy()
        if fov is not None:
            show = show[show["fovID"].fillna(-1).astype(int) == int(fov)]
        show = show[show["Session"].astype(str) == session]
        if show.empty:
            return
        session_root = self._resolve_session_root(session)
        if session_root is None:
            return
        sd = self.cache.get(session, session_root)

        # Candidate centroids
        ids = pd.to_numeric(show["suite2pID"], errors="coerce").fillna(-1).astype(int).to_numpy()
        pts = []
        row_inds = show.index.to_numpy()
        for ridx, roi_1b in zip(row_inds, ids):
            if roi_1b <= 0 or roi_1b - 1 >= len(sd.stat):
                continue
            x, y = _roi_centroid_from_stat(sd.stat[roi_1b - 1])
            if np.isfinite(x) and np.isfinite(y):
                pts.append((ridx, x, y))
        if not pts:
            return
        xy = np.asarray([(p[1], p[2]) for p in pts], dtype=float)
        click = np.asarray([event.xdata, event.ydata], dtype=float)
        d2 = np.sum((xy - click[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        best_row = int(pts[j][0])
        if float(np.sqrt(d2[j])) > 18:  # pixels
            return
        self._select_row(best_row, from_table=False)

    def _select_row(self, row_idx: int, from_table: bool):
        if self.t is None:
            return
        if self._in_select:
            return
        self._in_select = True
        t0 = time.time()
        try:
            self._selected_row = int(row_idx)

            cid = _safe_int(self.t.loc[row_idx, self.COL_CUR], None)
            if cid is None:
                cid = _safe_int(self.t.loc[row_idx, self.COL_ORIG], None)
            self._selected_cellid = cid

            # Switch view so user sees both sessions for this cellRegID
            if cid is not None:
                self.table_view_var.set("Selected cellRegID")

            # Rebuild table and redraw image
            self._populate_table()
            self._render()

            # Programmatic highlight WITHOUT triggering recursion storms
            self._suppress_table_event = True
            try:
                self._ignore_table_select += 1
                iid = str(row_idx)
                # only if item exists in current tree
                if iid in self.tree.get_children():
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
            finally:
                self._suppress_table_event = False

            # Optional: popup footprints across sessions for this curated/original cell id
            if cid is not None and bool(self.popups_var.get()):
                self.after(1, lambda c=int(cid): self._popup_footprints_for_cellid(c))

            self._log(f"selected row={row_idx} | curatedID={cid} dt={time.time()-t0:.3f}s")
        except Exception as e:
            self._log(f"select_row error: {e}")
            self._log(traceback.format_exc())
        finally:
            self._in_select = False

    def _popup_footprints_for_cellid(self, cid: int):
        """
        Show native-coordinates footprints for this curated/original ID across sessions.

        NEW (v3):
        - Click on an ROI (nearest centroid) in any session panel to select it as a candidate.
        - Then click "Link candidate → base" to merge it into the selected curatedID.
        - Reuses a single Tk Toplevel + embedded Matplotlib canvas (no endless plt.figure windows).
        """
        if self.t is None:
            return

        # Filter by current FOV if available
        fov = self.current_fov
        tt = self.t.copy()
        if fov is not None and "fovID" in tt.columns:
            tt = tt[pd.to_numeric(tt["fovID"], errors="coerce").fillna(-1).astype(int) == int(fov)]
        if tt.empty:
            return

        # Base curatedID is the thing we are curating against
        self._popup_base_cid = int(cid)
        self._popup_candidate = None

        # Session order: by UnixTime if available
        sessions = list(pd.unique(tt["Session"].astype(str)))
        if "UnixTime" in tt.columns:
            try:
                sess_ut = []
                for s in sessions:
                    sub = tt[tt["Session"].astype(str) == s]
                    ut = pd.to_numeric(sub["UnixTime"], errors="coerce")
                    sess_ut.append((s, float(np.nanmedian(ut)) if np.isfinite(np.nanmedian(ut)) else np.inf))
                sessions = [s for s, _ in sorted(sess_ut, key=lambda x: x[1])]
            except Exception:
                sessions = sorted(sessions)

        # Ensure popup window exists + build controls
        if self._popup_win is None or not bool(self._popup_win.winfo_exists()):
            win = tk.Toplevel(self)
            win.title("Footprints (native coords) — click to pick candidate")
            win.geometry("1120x920")
            self._popup_win = win

            # Controls
            ctrl = ttk.Frame(win)
            ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

            ttk.Label(ctrl, text="Candidate:").pack(side=tk.LEFT)
            lbl = ttk.Label(ctrl, textvariable=self._popup_status_var, width=90, font=("Segoe UI", 11))
            lbl.pack(side=tk.LEFT, padx=(6, 10))

            btn_link = ttk.Button(ctrl, text="Link candidate → base", command=self._popup_link_candidate)
            btn_link.pack(side=tk.LEFT, padx=(0, 6))

            btn_unlink = ttk.Button(ctrl, text="Unlink candidate (new curatedID)", command=self._popup_unlink_candidate)
            btn_unlink.pack(side=tk.LEFT, padx=(0, 6))

            # Selection tolerance + outline toggle
            ttk.Label(ctrl, text="Pick tol (px):").pack(side=tk.LEFT, padx=(12, 2))
            self._popup_tol_var = tk.StringVar(value=str(int(self._popup_pick_tol_px)))
            ent_tol = ttk.Entry(ctrl, textvariable=self._popup_tol_var, width=6)
            ent_tol.pack(side=tk.LEFT, padx=(0, 10))
            self._popup_show_outlines = tk.BooleanVar(value=True)
            ttk.Checkbutton(ctrl, text="Show ROI outlines", variable=self._popup_show_outlines, command=lambda: self._popup_footprints_for_cellid(int(self._popup_base_cid) if self._popup_base_cid is not None else cid)).pack(side=tk.LEFT, padx=(0, 10))
            self._popup_show_counts = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                ctrl,
                text="Show counts",
                variable=self._popup_show_counts,
                command=lambda: self._popup_footprints_for_cellid(int(self._popup_base_cid) if self._popup_base_cid is not None else cid),
            ).pack(side=tk.LEFT, padx=(0, 10))


            ttk.Button(ctrl, text="Clear selection", command=self._popup_clear_candidate).pack(side=tk.RIGHT, padx=(0, 8))
            ttk.Button(ctrl, text="Close", command=win.destroy).pack(side=tk.RIGHT)

            from matplotlib.figure import Figure
            fig = Figure(figsize=(10.8, 8.8), dpi=100)
            self._popup_fig = fig
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._popup_canvas = canvas

            # Connect click handler once
            def _close():
                try:
                    win.destroy()
                except Exception:
                    pass
                # cleanup
                self._popup_axes_to_session = {}
                self._popup_candidate = None
                self._popup_base_cid = None
                if self._popup_canvas is not None and self._popup_mpl_cid is not None:
                    try:
                        self._popup_canvas.mpl_disconnect(self._popup_mpl_cid)
                    except Exception:
                        pass
                self._popup_mpl_cid = None
                self._popup_win = None
                self._popup_fig = None
                self._popup_canvas = None

            win.protocol("WM_DELETE_WINDOW", _close)

            # mpl click binding
            try:
                self._popup_mpl_cid = canvas.mpl_connect("button_press_event", self._on_popup_click)
            except Exception:
                self._popup_mpl_cid = None

        # Build lookup: for each session, do we already have a row in this curatedID?
        cur = pd.to_numeric(tt[self.COL_CUR], errors="coerce").fillna(-1).astype(int)
        base_mask = (cur == int(cid))

        sess_to_row = {}
        for s in sessions:
            sub = tt[(tt["Session"].astype(str) == s) & base_mask]
            if len(sub) >= 1:
                sess_to_row[s] = int(sub.index[0])
            else:
                sess_to_row[s] = None

        # Render figure
        fig = self._popup_fig
        assert fig is not None
        fig.clear()
        self._popup_axes_to_session = {}

        n = len(sessions)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))

        axes = []
        for i, s in enumerate(sessions):
            ax = fig.add_subplot(rows, cols, i + 1)
            axes.append(ax)
            self._popup_axes_to_session[ax] = s

            # Load session data
            # SessionPath in table is the session folder root (not plane0)
            ss = tt[tt["Session"].astype(str) == s].iloc[0]
            sess_root = self._resolve_session_root(s)
            if sess_root is None:
                return
            sd = self.cache.get(s, sess_root)
            bg = _norm_img(sd.bg)

            ax.imshow(bg, cmap="gray", origin="upper", interpolation="nearest")
            ax.set_title(s, fontsize=13)
            ax.axis("off")
            # Context: paired vs unpaired (based on #sessions per id) + optional labels
            try:
                id_col = self.COL_CUR if (self.COL_CUR in tt.columns) else "cellRegID"
                tmp = tt.copy()
                tmp["_id"] = pd.to_numeric(tmp[id_col], errors="coerce").fillna(-1).astype(int)
                # counts: number of sessions each id appears in (within current FOV filter)
                id_to_n = tmp.groupby("_id")["Session"].nunique().to_dict()

                sub_sess = tmp[tmp["Session"].astype(str) == str(s)]
                suite_to_id = dict(
                    zip(
                        pd.to_numeric(sub_sess["suite2pID"], errors="coerce").fillna(-1).astype(int),
                        sub_sess["_id"].astype(int),
                    )
                )

                # centroids for all stat entries
                cents = []
                for st in sd.stat:
                    cx, cy = _roi_centroid_from_stat(st)
                    cents.append((cx, cy))
                cents = np.asarray(cents, dtype=float)

                keep = sd.iscell.astype(bool)
                if getattr(keep, "ndim", 1) > 1:
                    keep = keep[:, 0].astype(bool)
                idx_all = np.where(keep)[0]
                pts = cents[idx_all]

                # classify paired vs unpaired by count>=2
                paired_xy = []
                unpaired_xy = []
                labels = []  # (x,y,n)

                for rid0, (cx, cy) in zip(idx_all, pts):
                    roi_1b = int(rid0) + 1
                    _id = int(suite_to_id.get(roi_1b, -1))
                    n = int(id_to_n.get(_id, 1)) if _id != -1 else 1
                    if n >= 2:
                        paired_xy.append((cx, cy))
                    else:
                        unpaired_xy.append((cx, cy))
                    labels.append((cx, cy, n))

                if paired_xy:
                    arr = np.asarray(paired_xy, dtype=float)
                    ax.scatter(arr[:, 0], arr[:, 1], s=26, alpha=0.90, c="lime", edgecolors="black", linewidths=0.4)
                if unpaired_xy:
                    arr = np.asarray(unpaired_xy, dtype=float)
                    ax.scatter(arr[:, 0], arr[:, 1], s=26, alpha=0.82, c="orange", edgecolors="black", linewidths=0.4)

                # Optional counts labels
                try:
                    show_counts = bool(getattr(self, "_popup_show_counts").get())
                except Exception:
                    show_counts = False
                if show_counts:
                    for cx, cy, n in labels:
                        ax.text(
                            cx + 2,
                            cy + 2,
                            str(n),
                            color="white",
                            fontsize=9,
                            alpha=0.75,
                            bbox=dict(facecolor="black", alpha=0.15, edgecolor="none", pad=0.5),
                        )
            except Exception:
                pass
            except Exception:
                pass

            # Overlay the base cell if present in this session
            rid_text = "MISSING"
            row_idx = sess_to_row[s]
            if row_idx is not None:
                roi_1b = _safe_int(tt.loc[row_idx, "suite2pID"], None)
                if roi_1b is not None:
                    rid0 = roi_1b - 1
                    rid_text = f"suite2pID={roi_1b}"
                    if 0 <= rid0 < len(sd.stat):
                        poly = _roi_polygon_from_stat(sd.stat[rid0])
                        if poly is not None and poly.size:
                            ax.plot(poly[:, 0], poly[:, 1], color="cyan", linewidth=2.0)
                        # filled footprint overlay
                        fp = _roi_footprint_image_from_stat(sd.stat[rid0], shape_hw=sd.bg.shape[:2])
                        if fp is not None:
                            m = float(np.nanmax(fp)) if np.isfinite(np.nanmax(fp)) else 0.0
                            if m > 0:
                                ax.imshow(fp / m, cmap="viridis", alpha=0.45, origin="upper", interpolation="nearest")

            ax.text(0.02, 0.04, rid_text, transform=ax.transAxes,
                    fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"))

        fig.suptitle(f"Spatial footprints (native coords) | base curatedID {cid} | click to pick candidate", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Update status
        self._popup_status_var.set("Click an ROI in another session panel to select a candidate; then press Link. Centroids: green=paired (>=2 sessions), orange=unpaired (1 session).")
        self._popup_canvas.draw_idle()

    def _on_popup_click(self, event):
        """
        Matplotlib click handler for the popup window.
        Select ROI in the clicked session panel.

        Selection rule:
        1) If click is inside any iscell ROI polygon, pick that ROI.
        2) Otherwise pick nearest iscell centroid within tolerance.
        Toggle deselect: clicking the same ROI again clears selection.
        """
        if self.t is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        sess = self._popup_axes_to_session.get(ax, None)
        if sess is None:
            return
        sess = str(sess)

        # Update tolerance from UI if present
        try:
            if hasattr(self, "_popup_tol_var"):
                self._popup_pick_tol_px = float(self._popup_tol_var.get())
        except Exception:
            pass

        # Load session data
        tt = self.t
        if self.current_fov is not None and "fovID" in tt.columns:
            tt = tt[pd.to_numeric(tt["fovID"], errors="coerce").fillna(-1).astype(int) == int(self.current_fov)]
        sub = tt[tt["Session"].astype(str) == sess]
        if sub.empty:
            return
        sess_root = self._resolve_session_root(sess)
        if sess_root is None:
            return
        sd = self.cache.get(sess, sess_root)

        x = float(event.xdata); y = float(event.ydata)

        # Build index list for iscell
        keep = sd.iscell.astype(bool)
        if getattr(keep, "ndim", 1) > 1:
            keep = keep[:, 0].astype(bool)
        idx_all = np.where(keep)[0]
        if idx_all.size == 0:
            return

        picked_rid0 = None

        # 1) Inside-polygon test (fast enough for ~300-800 ROIs)
        try:
            from matplotlib.path import Path as MplPath
            for rid0 in idx_all:
                st = sd.stat[int(rid0)]
                xp = np.asarray(st.get("xpix", []), dtype=float)
                yp = np.asarray(st.get("ypix", []), dtype=float)
                if xp.size < 3 or yp.size < 3:
                    continue
                poly = np.stack([xp, yp], axis=1)
                # quick bbox reject
                if x < poly[:, 0].min() - 1 or x > poly[:, 0].max() + 1 or y < poly[:, 1].min() - 1 or y > poly[:, 1].max() + 1:
                    continue
                if MplPath(poly).contains_point((x, y)):
                    picked_rid0 = int(rid0)
                    break
        except Exception:
            picked_rid0 = None

        # 2) Nearest centroid fallback
        if picked_rid0 is None:
            cents = []
            for st in sd.stat:
                cx, cy = _roi_centroid_from_stat(st)
                cents.append((cx, cy))
            cents = np.asarray(cents, dtype=float)
            pts = cents[idx_all]
            d = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
            j = int(np.argmin(d))
            if float(d[j]) > float(self._popup_pick_tol_px):
                self._popup_status_var.set(f"{sess}: click closer to an ROI (min dist {float(d[j]):.1f}px)")
                return
            picked_rid0 = int(idx_all[j])

        roi_1b = int(picked_rid0 + 1)

        # Toggle deselect if clicking same ROI again
        if self._popup_candidate is not None:
            if str(self._popup_candidate.get("session")) == sess and int(self._popup_candidate.get("suite2pID")) == roi_1b:
                self._popup_clear_candidate()
                return

        # Locate this ROI's row in mouse_table
        rowmatch = tt[(tt["Session"].astype(str) == sess) & (pd.to_numeric(tt["suite2pID"], errors="coerce") == roi_1b)]
        if rowmatch.empty:
            self._popup_status_var.set(f"{sess}: picked suite2pID={roi_1b} but row not found in mouse_table")
            return
        row_idx = int(rowmatch.index[0])

        self._popup_candidate = {"session": sess, "suite2pID": int(roi_1b), "row_idx": int(row_idx)}
        self._popup_status_var.set(f"Candidate selected: {sess} suite2pID={roi_1b} (row={row_idx}). Click 'Link candidate → base'.")

        try:
            self._popup_draw_candidate_overlay()
        except Exception:
            pass

        except Exception:
            pass

    def _popup_draw_candidate_overlay(self):
        """
        Highlight candidate ROI selection (magenta) without accumulating overlays.
        Also supports toggle: if same ROI clicked again, clears selection.
        """
        if self._popup_candidate is None or self._popup_fig is None or self._popup_canvas is None:
            return
        cand = self._popup_candidate
        sess = str(cand["session"])
        roi_1b = int(cand["suite2pID"])
        rid0 = roi_1b - 1

        # Find corresponding axis
        ax = None
        for a, s in self._popup_axes_to_session.items():
            if str(s) == sess:
                ax = a
                break
        if ax is None:
            return

        # Remove previous highlight for this session (if any)
        prev = self._popup_highlight.get(sess, None)
        if prev is not None:
            try:
                prev.remove()
            except Exception:
                pass
            self._popup_highlight.pop(sess, None)

        # Load session data
        tt = self.t
        if self.current_fov is not None and "fovID" in tt.columns:
            tt = tt[pd.to_numeric(tt["fovID"], errors="coerce").fillna(-1).astype(int) == int(self.current_fov)]
        sub = tt[tt["Session"].astype(str) == sess]
        if sub.empty:
            return
        sess_root = self._resolve_session_root(sess)
        if sess_root is None:
            return
        sd = self.cache.get(sess, sess_root)

        if 0 <= rid0 < len(sd.stat):
            poly = _roi_polygon_from_stat(sd.stat[rid0])
            if poly is not None and poly.size:
                # draw highlight line; keep handle so we can remove later
                ln, = ax.plot(poly[:, 0], poly[:, 1], color="magenta", linewidth=2.4, alpha=0.95)
                self._popup_highlight[sess] = ln
        self._popup_canvas.draw_idle()

    
    def _popup_clear_candidate(self):
        """Clear current candidate selection + remove highlight overlays."""
        self._popup_candidate = None
        self._popup_status_var.set("Candidate cleared. Click an ROI in another session panel to select a candidate; then press Link.")
        # remove highlights
        for sess, art in list(getattr(self, "_popup_highlight", {}).items()):
            try:
                art.remove()
            except Exception:
                pass
            try:
                self._popup_highlight.pop(sess, None)
            except Exception:
                pass
        try:
            if self._popup_canvas is not None:
                self._popup_canvas.draw_idle()
        except Exception:
            pass

    def _popup_link_candidate(self):
        """
        Link candidate ROI to the popup base curatedID by editing self.t[self.COL_CUR].
        """
        if self.t is None or self._popup_candidate is None or self._popup_base_cid is None:
            return
        base = int(self._popup_base_cid)
        row_idx = int(self._popup_candidate["row_idx"])

        # Merge candidate's existing curated cluster into base
        cur = pd.to_numeric(self.t[self.COL_CUR], errors="coerce")
        cand_cur = _safe_int(cur.loc[row_idx], None)
        if cand_cur is None:
            cand_cur = base

        # If different, merge all cand_cur into base
        if int(cand_cur) != int(base):
            self.t.loc[cur.fillna(-1).astype(int) == int(cand_cur), self.COL_CUR] = int(base)

        # Set this row to base (idempotent)
        self.t.loc[row_idx, self.COL_CUR] = int(base)

        self._log(f"linked candidate row={row_idx} to curatedID={base} (merged {cand_cur}→{base})")
        # Refresh table + popup
        self._populate_table()
        self._render()
        self._popup_footprints_for_cellid(int(base))

    def _popup_unlink_candidate(self):
        """
        Unlink candidate ROI by assigning a new unique curatedID to that single row.
        """
        if self.t is None or self._popup_candidate is None:
            return
        row_idx = int(self._popup_candidate["row_idx"])

        # New curatedID = max+1
        cur = pd.to_numeric(self.t[self.COL_CUR], errors="coerce").fillna(-1).astype(int)
        new_id = int(cur.max() + 1)

        self.t.loc[row_idx, self.COL_CUR] = new_id
        self._log(f"unlinked candidate row={row_idx} to new curatedID={new_id}")

        self._populate_table()
        self._render()
        self._popup_footprints_for_cellid(int(new_id))


    def _compute_iou_async(self):
        if self.t is None:
            return
        if linear_sum_assignment is None:
            messagebox.showerror("Missing dependency", "scipy.optimize.linear_sum_assignment not available.")
            return
        # Run on a background thread so the GUI doesn't freeze.
        th = threading.Thread(target=self._compute_iou_worker, daemon=True)
        th.start()
        self._log("IoU worker started…")

    def _compute_iou_worker(self):
        """
        Compute an alternative registration using IoU+Hungarian for this FOV.
        Produces column self.COL_IOU with iou-based cluster IDs.
        """
        assert self.t is not None
        fov = self.current_fov
        tt = self.t.copy()
        if fov is not None:
            tt = tt[tt["fovID"].fillna(-1).astype(int) == int(fov)]

        # sessions
        sessions = tt["Session"].astype(str).unique().tolist()
        if len(sessions) < 2:
            self._worker_q.put(("log", "IoU: need >=2 sessions in this FOV."))
            return

        # Choose reference = session with most rows (cells)
        counts = {s: int((tt["Session"].astype(str) == s).sum()) for s in sessions}
        ref_session = max(counts.keys(), key=lambda s: counts[s])
        self._worker_q.put(("log", f"IoU: ref_session={ref_session} (rows={counts[ref_session]})"))

        # Parameters (hardcoded defaults; could be exposed later as UI)
        mask_thr = 0.15
        iou_thr = 0.10
        dist_thr_um = 8.0
        microns_per_pixel = 2.0
        beta = 0.25
        dist_thr_px = dist_thr_um / microns_per_pixel

        # Preload session data + roi lists for this fov
        session_rows: Dict[str, np.ndarray] = {}
        session_roi_1b: Dict[str, np.ndarray] = {}
        session_sd: Dict[str, SessionData] = {}

        for s in sessions:
            sub = tt[tt["Session"].astype(str) == s]
            if sub.empty:
                continue
            session_root = self._resolve_session_root(s)
            if session_root is None:
                continue
            sd = self.cache.get(s, session_root)
            session_sd[s] = sd
            rows = sub.index.to_numpy()
            roi = pd.to_numeric(sub["suite2pID"], errors="coerce").fillna(-1).astype(int).to_numpy()
            session_rows[s] = rows
            session_roi_1b[s] = roi

        # Build reference masks + centroids
        sd_ref = session_sd[ref_session]
        ref_rows = session_rows[ref_session]
        ref_roi = session_roi_1b[ref_session]

        ref_masks: List[np.ndarray] = []
        ref_centroids: np.ndarray = np.zeros((ref_roi.size, 2), dtype=float)
        for k, roi_1b in enumerate(ref_roi):
            rid0 = roi_1b - 1
            if roi_1b <= 0 or rid0 >= len(sd_ref.stat):
                ref_masks.append(np.zeros((0,), dtype=np.int32))
                ref_centroids[k, :] = np.nan
                continue
            ref_masks.append(_roi_mask_indices_from_stat(sd_ref.stat[rid0], sd_ref.shape_hw, mask_threshold=mask_thr))
            cx, cy = _roi_centroid_from_stat(sd_ref.stat[rid0])
            ref_centroids[k, :] = (cx, cy)

        # Initialize iou cluster ids: each ref roi gets its own cluster id (1..Nref)
        next_id = int(ref_roi.size) + 1
        iou_id_by_row: Dict[int, int] = {int(r): int(k + 1) for k, r in enumerate(ref_rows)}

        # Match each other session to ref via Hungarian
        for s in sessions:
            if s == ref_session:
                continue
            sd = session_sd[s]
            rows = session_rows[s]
            roi = session_roi_1b[s]
            n_other = int(roi.size)
            if n_other == 0:
                continue
            self._worker_q.put(("log", f"IoU: matching {s} (n={n_other}) → {ref_session} (n={ref_roi.size})"))

            other_masks: List[np.ndarray] = []
            other_centroids: np.ndarray = np.zeros((n_other, 2), dtype=float)
            for j, roi_1b in enumerate(roi):
                rid0 = roi_1b - 1
                if roi_1b <= 0 or rid0 >= len(sd.stat):
                    other_masks.append(np.zeros((0,), dtype=np.int32))
                    other_centroids[j, :] = np.nan
                    continue
                other_masks.append(_roi_mask_indices_from_stat(sd.stat[rid0], sd.shape_hw, mask_threshold=mask_thr))
                cx, cy = _roi_centroid_from_stat(sd.stat[rid0])
                other_centroids[j, :] = (cx, cy)

            # Build cost matrix
            n_ref = int(ref_roi.size)
            cost = np.full((n_ref, n_other), 1e6, dtype=float)

            # Fast-ish pairwise IoU gated by centroid distance
            for i in range(n_ref):
                if not np.isfinite(ref_centroids[i, 0]):
                    continue
                dx = other_centroids[:, 0] - ref_centroids[i, 0]
                dy = other_centroids[:, 1] - ref_centroids[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                near = np.where(np.isfinite(dist) & (dist <= dist_thr_px))[0]
                if near.size == 0:
                    continue
                for j in near:
                    iou = _iou_from_sorted_indices(ref_masks[i], other_masks[j])
                    # cost: maximize IoU, penalize distance
                    cost[i, j] = (1.0 - iou) + beta * (float(dist[j]) / float(dist_thr_px + 1e-9))

            # Hungarian
            r_ind, c_ind = linear_sum_assignment(cost)
            # Accept matches with iou >= threshold and finite cost
            matched_other = set()
            for i, j in zip(r_ind, c_ind):
                if cost[i, j] >= 1e5:
                    continue
                # recompute iou (cheap)
                iou = _iou_from_sorted_indices(ref_masks[int(i)], other_masks[int(j)])
                if iou < iou_thr:
                    continue
                # assign other row to ref cluster
                row_other = int(rows[int(j)])
                row_ref = int(ref_rows[int(i)])
                iou_id_by_row[row_other] = iou_id_by_row[row_ref]
                matched_other.add(row_other)

            # Unmatched others get new cluster IDs
            for r in rows:
                rr = int(r)
                if rr not in matched_other:
                    iou_id_by_row[rr] = next_id
                    next_id += 1

        # Write result to dataframe
        out = self.t.copy()
        out[self.COL_IOU] = np.nan
        for ridx, cid in iou_id_by_row.items():
            out.loc[int(ridx), self.COL_IOU] = int(cid)

        self._worker_q.put(("iou_done", out))

    def _poll_worker_queue(self):
        try:
            while True:
                typ, payload = self._worker_q.get_nowait()
                if typ == "log":
                    self._log(str(payload))
                elif typ == "iou_done":
                    assert isinstance(payload, pd.DataFrame)
                    self.t = payload
                    self._log("IoU complete. Use Tools → Use IoU as curated (or inspect via label_mode='curated').")
                    self._populate_table()
                    self._render()
        except queue.Empty:
            pass
        finally:
            self.after(80, self._poll_worker_queue)

    def _apply_iou_as_curated(self):
        if self.t is None:
            return
        if self.COL_IOU not in self.t.columns or self.t[self.COL_IOU].isna().all():
            messagebox.showwarning("No IoU result", "Run Tools → Compute IoU+Hungarian first.")
            return
        self.t[self.COL_CUR] = pd.to_numeric(self.t[self.COL_IOU], errors="coerce")
        self._log("curated IDs set from IoU result.")
        self._populate_table()
        self._render()

    def _reset_curated(self):
        if self.t is None:
            return
        self.t[self.COL_CUR] = pd.to_numeric(self.t[self.COL_ORIG], errors="coerce")
        self._log("curated IDs reset to original cellRegID.")
        self._populate_table()
        self._render()

    # ---- Manual curation ----

    def _get_selected_table_rows(self) -> List[int]:
        sel = self.tree.selection()
        rows = []
        for s in sel:
            try:
                rows.append(int(s))
            except Exception:
                continue
        return rows

    def _link_selected_rows(self):
        if self.t is None:
            return
        rows = self._get_selected_table_rows()
        if len(rows) < 2:
            messagebox.showinfo("Link", "Select 2+ rows in the table (Ctrl/Shift) to link.")
            return
        cur = pd.to_numeric(self.t.loc[rows, self.COL_CUR], errors="coerce")
        # choose a target id: smallest existing, else new
        existing = cur.dropna().astype(int).to_list()
        if existing:
            target = int(min(existing))
        else:
            target = int(pd.to_numeric(self.t[self.COL_CUR], errors="coerce").max(skipna=True) or 0) + 1
        self.t.loc[rows, self.COL_CUR] = target
        self._log(f"linked {len(rows)} row(s) → curatedID={target}")
        self._populate_table()
        self._render()

    def _unlink_selected_rows(self):
        if self.t is None:
            return
        rows = self._get_selected_table_rows()
        if len(rows) < 1:
            return
        next_id = int(pd.to_numeric(self.t[self.COL_CUR], errors="coerce").max(skipna=True) or 0) + 1
        for r in rows:
            self.t.loc[int(r), self.COL_CUR] = next_id
            next_id += 1
        self._log(f"unlinked {len(rows)} row(s) into new curated IDs.")
        self._populate_table()
        self._render()

    # ---- Save ----

    def _save_curated(self):
        if self.t is None:
            return
        default_dir = self.mouse_folder if self.mouse_folder else Path.cwd()
        out_csv = filedialog.asksaveasfilename(
            title="Save curated mouse_table as CSV",
            initialdir=str(default_dir),
            initialfile="mouse_table_curated.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not out_csv:
            return
        out_csv = Path(out_csv)
        out_pkl = out_csv.with_suffix(".pkl")

        self.t.to_csv(out_csv, index=False)
        self.t.to_pickle(out_pkl)
        
        # Also overwrite canonical outputs in the mouse folder (requested):
        #   mouse_table.csv, mouse_table.pkl, mouse_table.npy, mouse_table.mat
        try:
            out_dir = Path(self.mouse_folder) if self.mouse_folder else Path(out_csv).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Canonical CSV/PKL (so reloading the mouse folder picks up your curation)
            self.t.to_csv(out_dir / "mouse_table.csv", index=False)
            self.t.to_pickle(out_dir / "mouse_table.pkl")

            # NPY: save record array (allow_pickle for strings)
            rec = self.t.reset_index(drop=True).to_records(index=False)
            np.save(out_dir / "mouse_table.npy", rec, allow_pickle=True)

            # MAT: save as a struct-like dict of columns under key 'mouse_table'
            md = {}
            df = self.t.reset_index(drop=True)
            for col in df.columns:
                arr = df[col].to_numpy()
                if arr.dtype.kind in ("U", "S") or arr.dtype == object:
                    arr = arr.astype(object)
                md[str(col)] = arr
            scipy.io.savemat(out_dir / "mouse_table.mat", {"mouse_table": md}, do_compression=True)

            self._log(f"overwrote canonical mouse_table.* in: {out_dir}")
        except Exception as e:
            self._log(f"WARNING: could not overwrite canonical mouse_table.*: {e}")
        self._log(f"saved curated mouse_table: {out_csv.name} (+ {out_pkl.name})")
        try:
            out_dir2 = Path(self.mouse_folder) if self.mouse_folder else Path(out_csv).parent
            msg = (
                f"Saved curated table:\\n{out_csv}\\n\\n"
                f"Also overwrote canonical:\\n{out_dir2 / 'mouse_table.pkl'}"
            )
            messagebox.showinfo("Saved", msg)
        except Exception:
            pass


def launch():
    app = CellRegCurator()
    app.mainloop()


if __name__ == "__main__":
    launch()
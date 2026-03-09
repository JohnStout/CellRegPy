\
"""
viz_cellreg_alignment_gui_v8.py

Fixes the two issues you showed:
1) Mean image WAY too dark  -> now displayed with MATLAB-like percentile scaling (1..99%)
   (matches CellRegPy plotting helpers that use percentiles). fileciteturn62file8
2) Footprints appear rotated 90° -> caused by background image orientation mismatch.
   This version loads stat/ops preferentially from suite2p *.npy (canonical),
   and then auto-reorients the background to match the ROI coordinate system
   inferred from stat xpix/ypix (falls back to Fall.mat when needed).

Also adds:
- Robust loader: uses suite2p/plane0/stat.npy + ops.npy + iscell.npy when present,
  otherwise uses Fall.mat.
- Auto background reorientation to match ROI coordinate system:
  tries {identity, transpose, rot90/180/270 (+ transpose variants)} and picks the
  one whose shape matches the ROI coordinate system; if multiple match, chooses the
  one with highest corr vs a quick ROI-projection map.
- Status bar shows the chosen bg transform so you can immediately see what happened.

Run (Interactive / VSCode):
    from viz_cellreg_alignment_gui_v8 import launch
    launch()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import time
import threading
import traceback
import queue

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
from matplotlib.figure import Figure

from scipy.io import loadmat

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None


# ----------------------------- helpers -----------------------------

def _truncate(s: str, n: int = 34) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n]


def _norm01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=float)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def _imshow_gray(ax, img: np.ndarray, title: str = "", **kwargs):
    """MATLAB-ish display: percentile scaling."""
    img = np.asarray(img, dtype=float)
    lo, hi = np.nanpercentile(img, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest", **kwargs)
    if title:
        ax.set_title(title, fontsize=9)


def _find_plane0(session_root: Path) -> Path:
    """Return suite2p/plane0 folder given either session root or a deeper path.

    IMPORTANT: Avoid slow rglob on network/Dropbox drives by checking common paths first.
    """
    session_root = Path(session_root)

    # If they passed plane0 already
    if session_root.name.startswith("plane") and (session_root / "stat.npy").exists():
        return session_root

    # Fast/common paths (no recursion)
    direct = session_root / "suite2p" / "plane0"
    if (direct / "stat.npy").exists() or direct.exists():
        return direct

    direct2 = session_root / "plane0"
    if (direct2 / "stat.npy").exists() or direct2.exists():
        return direct2

    # Fallback: recursive search (can be slow on Dropbox)
    cands = list(session_root.rglob("suite2p/plane0"))
    if cands:
        cands.sort(key=lambda p: len(p.parts))
        return cands[0]
    cands = list(session_root.rglob("plane0"))
    if cands:
        cands.sort(key=lambda p: len(p.parts))
        return cands[0]

    raise FileNotFoundError(f"Could not find suite2p/plane0 under: {session_root}")

def _find_fall_mat(session_root: Path) -> Optional[Path]:
    try:
        p0 = _find_plane0(session_root)
    except Exception:
        return None
    fm = p0 / "Fall.mat"
    return fm if fm.exists() else None


def _load_from_npy(plane0: Path) -> Tuple[Dict[str, Any], List[Any], np.ndarray]:
    """Load ops/stat/iscell from suite2p npy files (preferred)."""
    ops_p = plane0 / "ops.npy"
    stat_p = plane0 / "stat.npy"
    iscell_p = plane0 / "iscell.npy"

    if not (ops_p.exists() and stat_p.exists()):
        raise FileNotFoundError("ops.npy/stat.npy missing")

    ops = np.load(ops_p, allow_pickle=True).item()
    stat = np.load(stat_p, allow_pickle=True)

    if iscell_p.exists():
        iscell = np.load(iscell_p, allow_pickle=True)
        if iscell.ndim == 2:
            iscell_b = iscell[:, 0].astype(bool)
        else:
            iscell_b = iscell.astype(bool)
    else:
        # fallback: treat all as cells
        iscell_b = np.ones(len(stat), dtype=bool)

    stat_list = list(stat.tolist()) if isinstance(stat, np.ndarray) else list(stat)
    return ops, stat_list, iscell_b


def _load_from_fallmat(plane0: Path) -> Tuple[Dict[str, Any], List[Any], np.ndarray, Dict[str, Any]]:
    """Load ops/stat/iscell (and traces dict) from Fall.mat."""
    fall_path = plane0 / "Fall.mat"
    if not fall_path.exists():
        raise FileNotFoundError(f"Fall.mat not found at {fall_path}")
    data = loadmat(str(fall_path), squeeze_me=True, struct_as_record=False)

    ops_obj = data.get("ops", None)
    ops: Dict[str, Any] = {}
    if ops_obj is not None:
        # mat_struct -> attrs
        try:
            for k in dir(ops_obj):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(ops_obj, k)
                except Exception:
                    continue
                if callable(v):
                    continue
                ops[k] = v
        except Exception:
            ops = {}

    stat_obj = data.get("stat", None)
    stat_list: List[Any] = []
    if isinstance(stat_obj, (list, tuple)):
        stat_list = list(stat_obj)
    elif isinstance(stat_obj, np.ndarray):
        if stat_obj.dtype == object:
            stat_list = list(stat_obj.flat)
        else:
            stat_list = [stat_obj[i] for i in range(stat_obj.shape[0])]

    iscell_obj = data.get("iscell", None)
    if isinstance(iscell_obj, np.ndarray):
        if iscell_obj.ndim == 2 and iscell_obj.shape[1] >= 1:
            iscell_b = iscell_obj[:, 0].astype(bool)
        else:
            iscell_b = iscell_obj.astype(bool)
    else:
        iscell_b = np.ones(len(stat_list), dtype=bool)

    traces = {}
    for k in ("F", "Fneu", "C", "S", "spks", "s2pSpk"):
        if k in data:
            traces[k] = np.asarray(data[k])

    return ops, stat_list, iscell_b, traces


def _roi_shape_from_stat(stat_list: List[Any]) -> Optional[Tuple[int, int]]:
    """Infer (H,W) = (Ly,Lx) from stat xpix/ypix."""
    maxx = -1
    maxy = -1
    minx = 1e9
    miny = 1e9
    seen = 0
    for st in stat_list[: min(len(stat_list), 400)]:
        try:
            if isinstance(st, dict):
                x = np.asarray(st.get("xpix"), dtype=float).ravel()
                y = np.asarray(st.get("ypix"), dtype=float).ravel()
            else:
                x = np.asarray(getattr(st, "xpix"), dtype=float).ravel() if hasattr(st, "xpix") else np.asarray(st["xpix"], dtype=float).ravel()
                y = np.asarray(getattr(st, "ypix"), dtype=float).ravel() if hasattr(st, "ypix") else np.asarray(st["ypix"], dtype=float).ravel()
            if x.size == 0:
                continue
            seen += 1
            maxx = max(maxx, int(np.nanmax(x)))
            maxy = max(maxy, int(np.nanmax(y)))
            minx = min(minx, float(np.nanmin(x)))
            miny = min(miny, float(np.nanmin(y)))
        except Exception:
            continue

    if seen == 0 or maxx < 0 or maxy < 0:
        return None

    # If looks 1-based (min>=1), convert to 0-based for shape inference
    if (minx >= 1.0) and (miny >= 1.0):
        maxx -= 1
        maxy -= 1

    H = int(maxy + 1)
    W = int(maxx + 1)
    if H <= 1 or W <= 1:
        return None
    return (H, W)


def _roi_projection_map(stat_list: List[Any], iscell: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Quick ROI projection for orientation scoring."""
    H, W = shape_hw
    proj = np.zeros((H, W), dtype=float)
    n = min(len(stat_list), len(iscell))
    take = np.where(np.asarray(iscell[:n]).astype(bool))[0]
    # subsample to keep fast
    if take.size > 250:
        rng = np.random.default_rng(0)
        take = rng.choice(take, size=250, replace=False)
    for i in take:
        st = stat_list[int(i)]
        try:
            if isinstance(st, dict):
                x = np.asarray(st.get("xpix"), dtype=int).ravel()
                y = np.asarray(st.get("ypix"), dtype=int).ravel()
            else:
                x = np.asarray(getattr(st, "xpix"), dtype=int).ravel() if hasattr(st, "xpix") else np.asarray(st["xpix"], dtype=int).ravel()
                y = np.asarray(getattr(st, "ypix"), dtype=int).ravel() if hasattr(st, "ypix") else np.asarray(st["ypix"], dtype=int).ravel()
            if x.size == 0:
                continue
            # handle 1-based
            if x.min(initial=0) >= 1 and y.min(initial=0) >= 1 and (x.max(initial=0) >= W or y.max(initial=0) >= H):
                x = x - 1
                y = y - 1
            x = np.clip(x, 0, W - 1)
            y = np.clip(y, 0, H - 1)
            proj[y, x] = 1.0
        except Exception:
            continue
    return proj


def _corr2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 50:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.dot(a, b) / den)


def _reorient_to_roi_shape(bg: np.ndarray, roi_shape: Tuple[int, int], proj: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """Try transforms on bg so that bg.shape == roi_shape."""
    bg = np.asarray(bg)
    candidates: List[Tuple[str, np.ndarray]] = [
        ("id", bg),
        ("T", bg.T),
        ("R90", np.rot90(bg, 1)),
        ("R180", np.rot90(bg, 2)),
        ("R270", np.rot90(bg, 3)),
        ("T_R90", np.rot90(bg.T, 1)),
        ("T_R180", np.rot90(bg.T, 2)),
        ("T_R270", np.rot90(bg.T, 3)),
    ]
    matches = [(name, im) for name, im in candidates if tuple(im.shape[:2]) == tuple(roi_shape)]
    if not matches:
        # can't match shape; keep as-is
        return bg, "id(no-shape-match)"
    if len(matches) == 1 or proj is None:
        return matches[0][1], matches[0][0]

    # choose best corr with ROI projection map (both normalized)
    proj_n = _norm01(proj)
    best = matches[0]
    best_sc = -np.inf
    for name, im in matches:
        sc = _corr2(_norm01(im), proj_n)
        if np.isfinite(sc) and sc > best_sc:
            best = (name, im)
            best_sc = sc
    return best[1], f"{best[0]}(corr={best_sc:.3f})"


def _roi_polygon_from_stat(stat_i: Any) -> Optional[np.ndarray]:
    try:
        if isinstance(stat_i, dict):
            x = np.asarray(stat_i.get("xpix"), dtype=float).ravel()
            y = np.asarray(stat_i.get("ypix"), dtype=float).ravel()
        else:
            if hasattr(stat_i, "xpix") and hasattr(stat_i, "ypix"):
                x = np.asarray(getattr(stat_i, "xpix"), dtype=float).ravel()
                y = np.asarray(getattr(stat_i, "ypix"), dtype=float).ravel()
            else:
                x = np.asarray(stat_i["xpix"], dtype=float).ravel()
                y = np.asarray(stat_i["ypix"], dtype=float).ravel()

        if x.size == 0:
            return None
        pts = np.vstack([x, y]).T
        if pts.shape[0] < 3 or ConvexHull is None:
            return pts
        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices]
        except Exception:
            return pts
    except Exception:
        return None


def _roi_footprint_image_from_stat(stat_i: Any, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    H, W = int(shape_hw[0]), int(shape_hw[1])
    try:
        if isinstance(stat_i, dict):
            x = np.asarray(stat_i.get("xpix"), dtype=int).ravel()
            y = np.asarray(stat_i.get("ypix"), dtype=int).ravel()
            lam = stat_i.get("lam", None)
        else:
            if hasattr(stat_i, "xpix") and hasattr(stat_i, "ypix"):
                x = np.asarray(getattr(stat_i, "xpix"), dtype=int).ravel()
                y = np.asarray(getattr(stat_i, "ypix"), dtype=int).ravel()
                lam = getattr(stat_i, "lam", None) if hasattr(stat_i, "lam") else None
            else:
                x = np.asarray(stat_i["xpix"], dtype=int).ravel()
                y = np.asarray(stat_i["ypix"], dtype=int).ravel()
                lam = None

        if x.size == 0:
            return None
        if lam is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(lam, dtype=float).ravel()
            if w.size != x.size:
                w = np.resize(w, x.size)

        # handle 1-based
        if (x.min(initial=0) >= 1) and (y.min(initial=0) >= 1) and (x.max(initial=0) >= W or y.max(initial=0) >= H):
            x = x - 1
            y = y - 1

        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        img = np.zeros((H, W), dtype=float)
        img[y, x] = w
        return img
    except Exception:
        return None


def _load_mouse_table_from_folder(mouse_folder: Path) -> pd.DataFrame:
    mouse_folder = Path(mouse_folder)
    pkl = mouse_folder / "mouse_table.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    csv = mouse_folder / "mouse_table.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Could not find mouse_table.pkl or mouse_table.csv in: {mouse_folder}")


def _load_mouse_table(mouse_folder: Optional[Path] = None, mouse_table_path: Optional[Path] = None) -> pd.DataFrame:
    if mouse_table_path is None:
        if mouse_folder is None:
            raise ValueError("Provide mouse_folder or mouse_table_path.")
        t = _load_mouse_table_from_folder(mouse_folder)
    else:
        mouse_table_path = Path(mouse_table_path)
        if mouse_table_path.suffix.lower() in (".pkl", ".pickle"):
            t = pd.read_pickle(mouse_table_path)
        elif mouse_table_path.suffix.lower() == ".csv":
            t = pd.read_csv(mouse_table_path)
        else:
            raise ValueError(f"Unsupported mouse_table type: {mouse_table_path}")

    required = {"Session", "suite2pID", "cellRegID", "SessionPath"}
    missing = required - set(t.columns)
    if missing:
        raise ValueError(f"mouse_table is missing columns: {sorted(missing)}")
    return t


def _safe_sort_rows_by_unix_time(df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    if "UnixTime" not in df.columns:
        return rows
    ser = pd.to_numeric(df.loc[rows, "UnixTime"], errors="coerce")
    if ser.isna().all():
        return rows
    vals = ser.to_numpy(dtype=float, copy=False)
    vals2 = np.where(np.isfinite(vals), vals, np.inf)
    order = np.argsort(vals2, kind="mergesort")
    return rows[order]


def _fov_scalar(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0]
    try:
        x = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if not np.isfinite(x):
            return None
        return int(x)
    except Exception:
        return None


# ----------------------------- session cache -----------------------------

@dataclass
class SessionData:
    session: str
    session_root: Path
    plane0: Path
    ops: Dict[str, Any]
    stat: List[Any]
    iscell: np.ndarray
    bg: np.ndarray
    bg_transform: str
    roi_shape: Tuple[int, int]
    fs: float
    traces: Dict[str, np.ndarray]


class SessionCache:
    """
    Thread-safe cache with optional async loading.

    - get(..., block=True) behaves like a normal cache (may block).
    - get(..., block=False) returns None if not yet loaded and kicks off a background load.
    """

    def __init__(self):
        self._cache: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._events: Dict[str, threading.Event] = {}
        self._log_fn = None  # optional callable

    def set_logger(self, fn):
        self._log_fn = fn

    def _log(self, msg: str):
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._events.clear()

    def preload_many(self, items: List[Tuple[str, Path]]):
        """Start background loads for many sessions."""
        for session, root in items:
            _ = self.get(session, root, block=False)

    def get(self, session: str, session_root: Path, block: bool = True) -> Optional[SessionData]:
        key = f"{session}::{Path(session_root)}"

        with self._lock:
            if key in self._cache:
                return self._cache[key]
            ev = self._events.get(key)
            if ev is None:
                ev = threading.Event()
                self._events[key] = ev
                # kick off background load
                th = threading.Thread(target=self._load_into_cache, args=(key, session, Path(session_root), ev), daemon=True)
                th.start()
            else:
                # already loading
                pass

        if not block:
            return None

        # Block until loaded
        ev.wait()
        with self._lock:
            return self._cache.get(key, None)

    def _load_into_cache(self, key: str, session: str, session_root: Path, ev: threading.Event):
        t0 = time.perf_counter()
        try:
            self._log(f"[cache] loading {session} | {session_root}")
            sd = self._load_session(session, session_root)
            dt = time.perf_counter() - t0
            with self._lock:
                self._cache[key] = sd
            self._log(f"[cache] loaded  {session} in {dt:.3f}s | plane0={sd.plane0} | bg={sd.bg_transform} | roi_shape={sd.roi_shape}")
        except Exception as e:
            dt = time.perf_counter() - t0
            self._log(f"[cache] ERROR loading {session} after {dt:.3f}s: {e}")
            traceback.print_exc()
        finally:
            try:
                ev.set()
            except Exception:
                pass

    def _load_session(self, session: str, session_root: Path) -> SessionData:
        session_root = Path(session_root)
        plane0 = _find_plane0(session_root)

        used = "npy"
        traces: Dict[str, np.ndarray] = {}

        t0 = time.perf_counter()
        try:
            ops, stat_list, iscell_b = _load_from_npy(plane0)
        except Exception:
            used = "Fall.mat"
            ops, stat_list, iscell_b, traces = _load_from_fallmat(plane0)
        self._log(f"[cache]   step ops/stat/iscell ({used}) took {time.perf_counter()-t0:.3f}s")

        # Always try to fetch traces from Fall.mat if present (even if ops/stat came from npy)
        if used == "npy":
            t1 = time.perf_counter()
            fm = _find_fall_mat(session_root)
            if fm is not None:
                try:
                    _, _, _, traces2 = _load_from_fallmat(plane0)
                    traces.update(traces2)
                except Exception:
                    pass
            self._log(f"[cache]   step traces(Fall.mat) took {time.perf_counter()-t1:.3f}s")

        # pick background image (meanImg preferred)
        bg = None
        bg_key = None
        t2 = time.perf_counter()
        for k in ("meanImg", "max_proj", "meanImgE"):
            if k in ops and ops[k] is not None:
                try:
                    bg = np.asarray(ops[k], dtype=float)
                    bg_key = k
                    break
                except Exception:
                    pass
        if bg is None:
            bg = np.zeros((512, 512), dtype=float)
            bg_key = "zeros"
        self._log(f"[cache]   step bg select ({bg_key}) took {time.perf_counter()-t2:.3f}s")

        # infer ROI coordinate system shape from stat
        t3 = time.perf_counter()
        roi_shape = _roi_shape_from_stat(stat_list)
        if roi_shape is None:
            roi_shape = tuple(bg.shape[:2])
        self._log(f"[cache]   step roi_shape took {time.perf_counter()-t3:.3f}s | roi_shape={roi_shape}")

        # build quick projection and reorient bg to match ROI coordinate system
        t4 = time.perf_counter()
        proj = _roi_projection_map(stat_list, iscell_b, roi_shape)
        self._log(f"[cache]   step roi_projection took {time.perf_counter()-t4:.3f}s")

        t5 = time.perf_counter()
        bg2, tf_name = _reorient_to_roi_shape(bg, roi_shape, proj=proj)
        self._log(f"[cache]   step reorient took {time.perf_counter()-t5:.3f}s | tf={tf_name}")

        # sanity: if bg still mismatched, pad/crop to roi_shape
        if tuple(bg2.shape[:2]) != tuple(roi_shape):
            H, W = roi_shape
            out = np.zeros((H, W), dtype=float)
            h0 = min(H, bg2.shape[0])
            w0 = min(W, bg2.shape[1])
            out[:h0, :w0] = bg2[:h0, :w0]
            bg2 = out
            tf_name = tf_name + "+padcrop"

        fs = float(ops.get("fs", np.nan)) if isinstance(ops, dict) else np.nan

        return SessionData(
            session=session,
            session_root=session_root,
            plane0=plane0,
            ops=ops,
            stat=stat_list,
            iscell=iscell_b,
            bg=bg2,
            bg_transform=f"{tf_name} (src={used}, bg={bg_key})",
            roi_shape=roi_shape,
            fs=fs if np.isfinite(fs) else np.nan,
            traces=traces,
        )

# ----------------------------- GUI -----------------------------

class CellRegViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CellRegPy Viewer (validation)")
        self.geometry("1500x880")

        self.cache = SessionCache()
        self._log_buffer: List[str] = []
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self.cache.set_logger(self._log)
        # poll log queue so background threads never touch Tk widgets
        self.after(100, self._flush_log_queue)
        self.mouse_folder: Optional[Path] = None

        self.t: Optional[pd.DataFrame] = None
        self.sessions: List[str] = []
        self.mouse_name: str = "mouse"
        self.current_session: Optional[str] = None

        self._suppress_table_event = False
        self._selected_row: Optional[int] = None
        self._selected_siblings: np.ndarray = np.array([], dtype=int)
        self._in_select: bool = False  # re-entrancy guard
        self.enable_popups = tk.BooleanVar(value=True)

        # footprint window state
        self._fp_win = None
        self._fp_fig = None
        self._fp_canvas = None
        self._fp_axes = None
        self._fp_queue = []
        self._fp_job = None

        self.show_cells_only = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.label_mode = tk.StringVar(value="both")
        self.table_view = tk.StringVar(value="Current session")

        self._build_widgets()

        self.fig = plt.Figure(figsize=(9.5, 7), dpi=100)
        self.ax_img = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        self._draw_empty()


    def _log(self, msg: str):
        """
        Thread-safe, timestamped logger.

        - Always prints to console.
        - Never touches Tk widgets from a background thread.
        - Background threads enqueue messages; main thread flushes via _flush_log_queue.
        """
        import time, threading
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass

        # If called from a worker thread, enqueue and return (Tk must be main-thread only).
        try:
            if threading.current_thread() is not threading.main_thread():
                try:
                    self._log_queue.put_nowait(line)
                except Exception:
                    pass
                return
        except Exception:
            # if threading isn't available, fall through
            pass

        # Main thread: append directly if debug box exists, else buffer.
        if hasattr(self, "debug_txt"):
            try:
                # flush any buffered lines first
                buf = getattr(self, "_log_buffer", [])
                if buf:
                    self.debug_txt.insert("end", "\n".join(buf) + "\n")
                    self._log_buffer = []
                self.debug_txt.insert("end", line + "\n")
                self.debug_txt.see("end")
            except Exception:
                pass
        else:
            try:
                self._log_buffer.append(line)
            except Exception:
                pass

    def _flush_log_queue(self):
        """
        Periodically flush background-thread log messages into the debug text box.
        Runs on the Tk main thread.
        """
        try:
            if hasattr(self, "debug_txt"):
                # flush buffer first
                buf = getattr(self, "_log_buffer", [])
                if buf:
                    self.debug_txt.insert("end", "\n".join(buf) + "\n")
                    self._log_buffer = []
                # flush queued messages
                drained = 0
                while True:
                    try:
                        line = self._log_queue.get_nowait()
                    except Exception:
                        break
                    self.debug_txt.insert("end", line + "\n")
                    drained += 1
                    if drained > 250:
                        # don't monopolize UI loop
                        break
                if drained:
                    self.debug_txt.see("end")
        except Exception:
            pass
        finally:
            try:
                self.after(100, self._flush_log_queue)
            except Exception:
                pass

    def _build_widgets(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Mouse Folder...", command=self._open_mouse_folder)
        file_menu.add_command(label="Open Mouse Table...", command=self._open_mouse_table)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.lbl_mouse = ttk.Label(top, text="Mouse: (no table loaded)")
        self.lbl_mouse.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top, text="Session:").pack(side=tk.LEFT)
        self.session_cb = ttk.Combobox(top, values=[], width=44, state="readonly")
        self.session_cb.pack(side=tk.LEFT, padx=6)
        self.session_cb.bind("<<ComboboxSelected>>", self._on_session_change)

        ttk.Checkbutton(top, text="Cells only (iscell)", variable=self.show_cells_only, command=self.refresh).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(top, text="Labels", variable=self.show_labels, command=self.refresh).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(top, text="Popups", variable=self.enable_popups).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Label mode:").pack(side=tk.LEFT, padx=(10, 0))
        self.label_cb = ttk.Combobox(top, values=["suite2pID", "cellRegID", "both"], width=10, state="readonly")
        self.label_cb.set("both")
        self.label_cb.pack(side=tk.LEFT, padx=6)
        self.label_cb.bind("<<ComboboxSelected>>", self._on_label_mode)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.plot_frame = ttk.Frame(main)
        main.add(self.plot_frame, weight=3)

        right = ttk.Frame(main)
        main.add(right, weight=2)

        hdr = ttk.Frame(right)
        hdr.pack(side=tk.TOP, fill=tk.X)
        self.tbl_title = ttk.Label(hdr, text="mouse_table")
        self.tbl_title.pack(side=tk.LEFT, anchor="w")

        ttk.Label(hdr, text="Table view:").pack(side=tk.LEFT, padx=(12, 0))
        self.table_view_cb = ttk.Combobox(hdr, values=["Current session", "Selected cell (siblings)"], width=22, state="readonly")
        self.table_view_cb.set("Current session")
        self.table_view_cb.pack(side=tk.LEFT, padx=6)
        self.table_view_cb.bind("<<ComboboxSelected>>", self._on_table_view_change)

        cols = ("row", "Session", "suite2pID", "cellRegID", "fovID", "UnixTime", "iscell")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=20)
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.column("row", width=55, anchor="center")
        self.tree.column("Session", width=170, anchor="w")
        self.tree.column("suite2pID", width=70, anchor="center")
        self.tree.column("cellRegID", width=70, anchor="center")
        self.tree.column("fovID", width=55, anchor="center")
        self.tree.column("UnixTime", width=95, anchor="center")
        self.tree.column("iscell", width=55, anchor="center")

        vsb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_table_select)

        dbg_frame = ttk.Frame(right)
        dbg_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, pady=(8, 0))
        ttk.Label(dbg_frame, text="Selection debug (siblings used):").pack(side=tk.TOP, anchor="w")
        self.debug_txt = tk.Text(dbg_frame, height=8, wrap="none")
        self.debug_txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # flush any buffered log lines now that debug box exists
        try: self._log('debug console ready')
        except Exception: pass

        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status = ttk.Label(bottom, text="File → Open Mouse Folder/Table to begin.")
        self.status.pack(side=tk.LEFT)

    def _open_mouse_folder(self):
        folder = filedialog.askdirectory(title="Select mouse folder (contains mouse_table.pkl/csv)")
        if not folder:
            return
        folder_p = Path(folder)
        t = _load_mouse_table(mouse_folder=folder_p)
        self.mouse_folder = folder_p
        self.set_table(t)

    def _open_mouse_table(self):
        f = filedialog.askopenfilename(title="Select mouse_table.pkl or mouse_table.csv", filetypes=[("Mouse table", "*.pkl *.pickle *.csv"), ("All files", "*.*")])
        if not f:
            return
        t = _load_mouse_table(mouse_table_path=Path(f))
        self.mouse_folder = Path(f).parent
        self.set_table(t)

    def _on_session_change(self, _evt=None):
        if not self.sessions:
            return
        self.current_session = self.sessions[int(self.session_cb.current())]
        self.table_view.set("Current session")
        self.table_view_cb.set("Current session")
        self.refresh()

    def _on_label_mode(self, _evt=None):
        self.label_mode.set(self.label_cb.get())
        self.refresh()

    def _on_table_view_change(self, _evt=None):
        self.table_view.set(self.table_view_cb.get())
        self._populate_tree()

    def _on_table_select(self, _evt=None):
        if self._suppress_table_event or self.t is None:
            return
        sel = self.tree.selection()
        if not sel:
            return
        try:
            row_idx = int(sel[0])
        except Exception:
            return
        self._select_row(row_idx, open_popups=True)

    def set_table(self, mouse_table: pd.DataFrame):
        self.cache.clear()
        t = mouse_table.copy()
        if "UnixTime" in t.columns:
            t["_UnixTime_num"] = pd.to_numeric(t["UnixTime"], errors="coerce")
            t = t.sort_values(["_UnixTime_num", "Session"], kind="mergesort").drop(columns=["_UnixTime_num"], errors="ignore")
        self.t = t
        self.sessions = list(pd.unique(t["Session"]))
        self.mouse_name = str(pd.unique(t["MouseName"])[0]) if "MouseName" in t.columns else "mouse"
        self.current_session = self.sessions[0] if self.sessions else None
        self.lbl_mouse.configure(text=f"Mouse: {self.mouse_name}")
        self.session_cb.configure(values=[_truncate(s, 50) for s in self.sessions])
        if self.sessions:
            self.session_cb.current(0)
        self.refresh()

        # Kick off background preloading for all sessions (keeps ROI-click fast).
        try:
            uniq = list(pd.unique(self.t["Session"]))
            items = []
            for s in uniq:
                # use the first SessionPath for that session
                rr = self.t.index[self.t["Session"] == s].to_numpy()
                if rr.size:
                    items.append((str(s), Path(str(self.t.loc[rr[0], "SessionPath"])) ))
            self._log(f"[preload] starting background preload for {len(items)} session(s)")
            self.cache.preload_many(items)
        except Exception as e:
            self._log(f"[preload] failed to start preload: {e}")


    def refresh(self):
        if self.t is None or self.current_session is None:
            self._draw_empty()
            return
        self._draw_session_image_and_rois()
        self._populate_tree()
        self.canvas.draw_idle()

    def _draw_empty(self):
        self.ax_img.cla()
        self.ax_img.set_axis_off()
        self.ax_img.text(0.5, 0.5, "File → Open Mouse Folder/Table", ha="center", va="center", transform=self.ax_img.transAxes)
        self.canvas.draw_idle()

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        self.debug_txt.delete("1.0", tk.END)
        if self.t is None or self.current_session is None:
            return

        if self.table_view.get() == "Selected cell (siblings)" and self._selected_siblings.size > 0:
            rows = self._selected_siblings
            title = "mouse_table — siblings"
        else:
            rows = self.t.index[self.t["Session"] == self.current_session].to_numpy()
            title = f"mouse_table — session: {self.current_session}"

        missing_any = False

        for r in rows:
            sess = str(self.t.loc[r, "Session"])
            sess_path = Path(str(self.t.loc[r, "SessionPath"]))
            suite2p = int(self.t.loc[r, "suite2pID"])
            cellreg = int(self.t.loc[r, "cellRegID"])
            fov = self.t.loc[r, "fovID"] if "fovID" in self.t.columns else ""
            ut = self.t.loc[r, "UnixTime"] if "UnixTime" in self.t.columns else ""
            iscell = ""
            try:
                sd = self.cache.get(sess, sess_path, block=False)
                if sd is None:
                    missing_any = True
                # suite2pID in table is 1-based; sd.iscell is 0-based
                rid0 = suite2p - 1
                if 0 <= rid0 < sd.iscell.size:
                    iscell = "1" if bool(sd.iscell[rid0]) else "0"
            except Exception:
                pass
            self.tree.insert("", "end", iid=str(int(r)), values=(int(r), sess, suite2p, cellreg, fov, ut, iscell))

        self.tbl_title.configure(text=f"{title} | rows={len(rows)}")

        if missing_any:
            # some sessions are still loading; refresh table soon
            self.after(200, self._populate_tree)

        if self._selected_row is not None and str(int(self._selected_row)) in self.tree.get_children():
            self._suppress_table_event = True
            try:
                self.tree.selection_set(str(int(self._selected_row)))
                self.tree.see(str(int(self._selected_row)))
            finally:
                self._suppress_table_event = False

    def _draw_session_image_and_rois(self):
        assert self.t is not None and self.current_session is not None
        self.ax_img.cla()
        self.ax_img.set_axis_off()

        rows = self.t.index[self.t["Session"] == self.current_session].to_numpy()
        if rows.size == 0:
            return

        session_root = Path(str(self.t.loc[rows[0], "SessionPath"]))
        sd = self.cache.get(self.current_session, session_root, block=False)
        if sd is None:
            # session still loading in background
            self.ax_img.text(0.5, 0.5, f"Loading {self.current_session}...", ha='center', va='center', transform=self.ax_img.transAxes)
            self.status.configure(text=f"Loading session {self.current_session}...")
            # retry soon
            self.after(100, self.refresh)
            return
        bg = np.asarray(sd.bg, dtype=float)

        _imshow_gray(self.ax_img, bg, title=f"{self.current_session} | bg={sd.bg_transform}")
        self.ax_img.set_aspect("equal")

        for r_i in rows:
            roi_id_1based = int(self.t.loc[r_i, "suite2pID"])
            rid0 = roi_id_1based - 1
            if rid0 < 0 or rid0 >= len(sd.stat):
                continue
            if self.show_cells_only.get():
                if not (0 <= rid0 < sd.iscell.size and bool(sd.iscell[rid0])):
                    continue
            poly = _roi_polygon_from_stat(sd.stat[rid0])
            if poly is None or poly.size == 0:
                continue
            patch = Polygon(poly, closed=True, fill=False, edgecolor=(0.2, 0.6, 1.0, 0.9), linewidth=0.8, picker=True)
            patch._cellreg_row_index = int(r_i)  # type: ignore[attr-defined]
            self.ax_img.add_patch(patch)

            if self.show_labels.get():
                cellreg = int(self.t.loc[r_i, "cellRegID"])
                if self.label_mode.get() == "suite2pID":
                    label = str(roi_id_1based)
                elif self.label_mode.get() == "cellRegID":
                    label = str(cellreg)
                else:
                    label = f"{roi_id_1based}|{cellreg}"
                self.ax_img.text(float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1])), label, color="yellow", fontsize=7)

        self.ax_img.set_xlim([0, bg.shape[1]])
        self.ax_img.set_ylim([bg.shape[0], 0])

        self.status.configure(text=f"Session {self.current_session} | bg_transform={sd.bg_transform} | roi_shape={sd.roi_shape}")

    def _on_pick(self, event):
        if self.t is None:
            return
        row_idx = getattr(event.artist, "_cellreg_row_index", None)
        if row_idx is None:
            return
        self._select_row(int(row_idx), open_popups=True)

    def _select_row(self, row_idx: int, open_popups: bool = True):
        assert self.t is not None
        if row_idx not in self.t.index:
            return
        if self._in_select:
            self._log(f"[ui] select_row reentry blocked row={row_idx}")
            return

        import time
        t0 = time.perf_counter()
        self._in_select = True
        try:
            self._log(f"[ui] select_row start row={row_idx}")
            self._selected_row = int(row_idx)

            cid = int(self.t.loc[row_idx, "cellRegID"])
            suite2p_sel = int(self.t.loc[row_idx, "suite2pID"])
            fov_val = _fov_scalar(self.t.loc[row_idx, "fovID"]) if "fovID" in self.t.columns else None

            # siblings
            t1 = time.perf_counter()
            if fov_val is None:
                sib_mask = (self.t["cellRegID"].astype(int) == cid)
            else:
                sib_mask = (self.t["cellRegID"].astype(int) == cid) & (pd.to_numeric(self.t["fovID"], errors="coerce") == fov_val)
            siblings = self.t.index[sib_mask].to_numpy()
            siblings = _safe_sort_rows_by_unix_time(self.t, siblings)
            self._selected_siblings = siblings
            self._log(f"[ui] siblings computed n={int(siblings.size)} dt={(time.perf_counter()-t1):.3f}s")

            # Switch table view to siblings automatically
            self.table_view.set("Selected cell (siblings)")
            self.table_view_cb.set("Selected cell (siblings)")

            t2 = time.perf_counter()
            self._populate_tree()
            self._log(f"[ui] table populated dt={(time.perf_counter()-t2):.3f}s")

            # programmatic selection without recursion
            self._suppress_table_event = True
            try:
                if str(int(row_idx)) in self.tree.get_children():
                    self.tree.selection_set(str(int(row_idx)))
                    self.tree.see(str(int(row_idx)))
            finally:
                self._suppress_table_event = False

            # debug panel (limit)
            self.debug_txt.delete("1.0", tk.END)
            self.debug_txt.insert(tk.END, f"Selected row={row_idx}\n")
            self.debug_txt.insert(tk.END, f"cellRegID={cid} | suite2pID(1-based)={suite2p_sel} | fovID={fov_val}\n\n")
            for r in siblings[:80]:
                self.debug_txt.insert(tk.END, f"row={int(r)} | session={self.t.loc[r,'Session']} | suite2pID={int(self.t.loc[r,'suite2pID'])}\n")
            if siblings.size > 80:
                self.debug_txt.insert(tk.END, f"... ({int(siblings.size)-80} more)\n")

            self.status.configure(text=f"Selected cellRegID={cid} | suite2pID={suite2p_sel} | siblings={int(siblings.size)}")

            # schedule popup (do not run inside callback)
            if open_popups and bool(self.enable_popups.get()) and siblings.size > 0:
                self._log("[ui] scheduling footprint render")
                sib_copy = np.array(siblings, copy=True)
                self.after(1, lambda r=sib_copy, c=cid: self._show_footprints(r, c))

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"[ui] ERROR in select_row: {e}")
        finally:
            self._in_select = False
            self._log(f"[ui] select_row end dt={(time.perf_counter()-t0):.3f}s")



    def _ensure_fp_window(self):
        """Create (or reuse) the non-blocking footprint Toplevel window."""
        if self._fp_win is not None:
            try:
                if int(self._fp_win.winfo_exists()) == 1:
                    return
            except Exception:
                pass

        self._fp_win = tk.Toplevel(self)
        self._fp_win.title("Footprints (native coords)")
        self._fp_win.geometry("1100x800")

        self._fp_fig = Figure(figsize=(10, 8), dpi=100)
        self._fp_canvas = FigureCanvasTkAgg(self._fp_fig, master=self._fp_win)
        self._fp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _on_close():
            try:
                if self._fp_job is not None:
                    self.after_cancel(self._fp_job)
            except Exception:
                pass
            self._fp_job = None
            self._fp_queue = []
            try:
                self._fp_win.destroy()
            except Exception:
                pass
            self._fp_win = None
            self._fp_fig = None
            self._fp_canvas = None

        self._fp_win.protocol("WM_DELETE_WINDOW", _on_close)

    def _show_footprints(self, rows: np.ndarray, cid: int):
        """
        Render footprints in a Tk Toplevel window without blocking the main GUI.

        Rendering is incremental (one subplot per Tk 'after' tick) so the GUI stays responsive
        even if we need to load a session from disk the first time.
        """
        assert self.t is not None
        n = int(rows.size)
        if n <= 0:
            self.configure(cursor="")
            return

        self._log(f"[render] show_footprints: n={n} (cid={cid})")
        self._ensure_fp_window()
        assert self._fp_fig is not None and self._fp_canvas is not None

        # clear old contents
        self._fp_fig.clf()

        cols = max(1, int(np.ceil(np.sqrt(n))))
        nrows = int(np.ceil(n / cols))

        axes = []
        for i in range(nrows * cols):
            ax = self._fp_fig.add_subplot(nrows, cols, i + 1)
            ax.axis("off")
            axes.append(ax)
        self._fp_axes = axes

        # queue the panels we need to render
        self._fp_queue = [(axes[i], int(rows[i])) for i in range(n)]
        self._fp_fig.suptitle(f"Spatial footprints (native coords) | CellRegID {cid}", fontsize=12, fontweight="bold")

        # cancel prior job if any
        if self._fp_job is not None:
            try:
                self.after_cancel(self._fp_job)
            except Exception:
                pass
            self._fp_job = None

        self._fp_job = self.after(1, lambda: self._render_fp_next(cid))

    def _render_fp_next(self, cid: int):
        """Render one panel and reschedule until done."""
        t0 = time.perf_counter()
        if self._fp_win is None or self._fp_fig is None or self._fp_canvas is None:
            self.configure(cursor="")
            return

        if not self._fp_queue:
            # finished
            try:
                self._fp_fig.tight_layout(rect=[0, 0, 1, 0.95])
            except Exception:
                pass
            self._fp_canvas.draw_idle()
            self._fp_job = None
            self.configure(cursor="")
            self._log(f"[render] finished footprints for cid={cid}")
            self.status.configure(text=f"Rendered footprints for CellRegID {cid}.")
            return

        ax, r = self._fp_queue.pop(0)

        try:
            sess = str(self.t.loc[r, "Session"])
            sess_root = Path(str(self.t.loc[r, "SessionPath"]))
            roi_1b = int(self.t.loc[r, "suite2pID"])
            rid0 = roi_1b - 1  # assumes mouse_table suite2pID is 1-based

            t_load0 = time.perf_counter()
            sd = self.cache.get(sess, sess_root, block=False)
            if sd is None:
                ax.text(0.5, 0.5, f"Loading {sess}...", ha='center', va='center', transform=ax.transAxes)
                # requeue and try again later
                self._fp_queue.append((ax, r))
                self._fp_canvas.draw_idle()
                self._fp_job = self.after(50, lambda: self._render_fp_next(cid))
                return
            self._log(f"[render] cache ready for {sess} in {time.perf_counter()-t_load0:.3f}s")
            bg = np.asarray(sd.bg, dtype=float)

            _imshow_gray(ax, bg, title=f"{_truncate(sess, 22)} | suite2pID={roi_1b}\n{sd.bg_transform}")
            if 0 <= rid0 < len(sd.stat):
                fp = _roi_footprint_image_from_stat(sd.stat[rid0], shape_hw=bg.shape[:2])
                if fp is not None:
                    m = float(np.nanmax(fp)) if np.isfinite(np.nanmax(fp)) else 0.0
                    if m > 0:
                        ax.imshow(fp / m, cmap="viridis", alpha=0.70, origin="upper", interpolation="nearest")
                poly = _roi_polygon_from_stat(sd.stat[rid0])
                if poly is not None and poly.size:
                    ax.plot(poly[:, 0], poly[:, 1], color="cyan", linewidth=1.4)
            ax.set_aspect("equal")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", transform=ax.transAxes)

        self._fp_canvas.draw_idle()
        self._log(f"[render] drew one panel in {time.perf_counter()-t0:.3f}s; remaining={len(self._fp_queue)}")
        self._fp_job = self.after(1, lambda: self._render_fp_next(cid))




def launch():
    app = CellRegViewer()
    app.mainloop()


if __name__ == "__main__":
    launch()
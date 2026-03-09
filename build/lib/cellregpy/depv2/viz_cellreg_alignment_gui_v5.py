# -*- coding: utf-8 -*-
"""
viz_cellreg_alignment_gui_v5.py

CellRegPy validation GUI:
- Loads mouse_table (from mouse folder) and per-session Suite2p Fall.mat
- Click ROI -> shows (a) footprint heatmaps across sessions + (b) CellReg-aligned overlay (if registration_results.npy present)
- Fixes UnixTime=None sorting crash by using pandas numeric sorting.

Run (VS Code Interactive / Jupyter):
    from viz_cellreg_alignment_gui_v5 import launch
    launch()

Or terminal:
    python viz_cellreg_alignment_gui_v5.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

from scipy.io import loadmat

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

try:
    from skimage import transform as sktransform
except Exception:
    sktransform = None


def _find_fall_mat(session_path: Path) -> Path:
    session_path = Path(session_path)
    cands = list(session_path.rglob("suite2p/plane0/Fall.mat"))
    if not cands:
        if (session_path / "Fall.mat").exists():
            return session_path / "Fall.mat"
        raise FileNotFoundError(f"Could not find suite2p/plane0/Fall.mat under: {session_path}")
    cands.sort(key=lambda p: len(p.parts))
    return cands[0]


def _ops_to_dict(ops: Any) -> Dict[str, Any]:
    if ops is None:
        return {}
    if isinstance(ops, dict):
        return ops
    out: Dict[str, Any] = {}
    for k in ("max_proj", "meanImgE", "meanImg", "Ly", "Lx", "fs"):
        try:
            if hasattr(ops, k):
                out[k] = getattr(ops, k)
        except Exception:
            pass
    return out


def _stat_to_list(stat: Any) -> List[Any]:
    if stat is None:
        return []
    if isinstance(stat, (list, tuple)):
        return list(stat)
    if isinstance(stat, np.ndarray):
        if stat.dtype == object:
            return list(stat.flat)
        return [stat[i] for i in range(stat.shape[0])]
    return []


def _iscell_to_boolvec(iscell: Any, n_rois: int) -> np.ndarray:
    if iscell is None:
        return np.ones(n_rois, dtype=bool)
    arr = np.asarray(iscell)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        return arr[:, 0].astype(bool)
    return arr.astype(bool)


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
                lam = stat_i["lam"] if hasattr(stat_i, "dtype") and stat_i.dtype.names and "lam" in stat_i.dtype.names else None

        if x.size == 0:
            return None

        w = np.ones_like(x, dtype=float) if lam is None else np.asarray(lam, dtype=float).ravel()
        if w.size != x.size:
            w = np.resize(w, x.size)

        # heuristic for 1-based
        if x.max(initial=0) >= W or y.max(initial=0) >= H:
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


def _safe_sort_rows_by_unix_time(df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    if "UnixTime" not in df.columns:
        return rows
    ser = pd.to_numeric(df.loc[rows, "UnixTime"], errors="coerce")
    if ser.isna().all():
        return rows
    vals = ser.to_numpy(dtype=float, copy=False)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    return rows[np.argsort(vals, kind="mergesort")]


def _load_fov_regs(mouse_folder: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    root = Path(mouse_folder) / "1_CellReg"
    if not root.exists():
        return out
    for fov_dir in sorted([p for p in root.glob("FOV*") if p.is_dir()]):
        reg_path = fov_dir / "Results" / "registration_results.npy"
        if not reg_path.exists():
            continue
        try:
            reg = np.load(reg_path, allow_pickle=True).item()
        except Exception:
            continue
        try:
            fov_id = int(fov_dir.name.replace("FOV", ""))
        except Exception:
            continue
        out[fov_id] = reg
    return out


def _guess_session_label_from_path(p: str, known_sessions: List[str]) -> str:
    try:
        P = Path(p)
        for parent in P.parents:
            if parent.name in known_sessions:
                return parent.name
    except Exception:
        pass
    return Path(p).stem


@dataclass
class SessionData:
    session: str
    session_path: Path
    fall_path: Path
    ops: Dict[str, Any]
    stat: List[Any]
    iscell: np.ndarray
    bg: np.ndarray
    fs: float
    F: Optional[np.ndarray] = None
    Fneu: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    spks: Optional[np.ndarray] = None


class SessionCache:
    def __init__(self):
        self._cache: Dict[str, SessionData] = {}

    def clear(self):
        self._cache.clear()

    def get(self, session: str, session_path: Path) -> SessionData:
        key = f"{session}::{session_path}"
        if key in self._cache:
            return self._cache[key]
        sd = self._load(session, session_path)
        self._cache[key] = sd
        return sd

    def _load(self, session: str, session_path: Path) -> SessionData:
        fall_path = _find_fall_mat(session_path)
        data = loadmat(str(fall_path), squeeze_me=True, struct_as_record=False)
        ops = _ops_to_dict(data.get("ops", None))
        stat = _stat_to_list(data.get("stat", None))
        iscell = _iscell_to_boolvec(data.get("iscell", None), len(stat))

        bg = None
        for k in ("max_proj", "meanImgE", "meanImg"):
            if k in ops and ops[k] is not None:
                try:
                    bg = np.asarray(ops[k], dtype=float)
                    break
                except Exception:
                    pass
        if bg is None:
            bg = np.zeros((512, 512), dtype=float)

        fs = float(ops.get("fs", np.nan))
        if not np.isfinite(fs) or fs <= 0:
            fs = np.nan

        def grab(name: str) -> Optional[np.ndarray]:
            v = data.get(name, None)
            return None if v is None else np.asarray(v)

        F = grab("F")
        Fneu = grab("Fneu")
        C = grab("C")
        S = grab("S")
        spks = grab("spks")
        if spks is None:
            spks = grab("s2pSpk")

        return SessionData(session, Path(session_path), fall_path, ops, stat, iscell, bg, fs, F, Fneu, C, S, spks)


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CellRegPy Viewer (validation)")
        self.geometry("1250x820")

        self.cache = SessionCache()
        self.mouse_folder: Optional[Path] = None
        self.fov_regs: Dict[int, Dict[str, Any]] = {}

        self.t: Optional[pd.DataFrame] = None
        self.sessions: List[str] = []
        self.current_session: Optional[str] = None

        self.show_cells_only = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.trace_kind = tk.StringVar(value="F (Suite2p)")
        self.show_fneu = tk.BooleanVar(value=False)
        self.show_spikes = tk.BooleanVar(value=True)

        self._build_widgets()

        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        self.ax_img = self.fig.add_subplot(2, 1, 1)
        self.ax_tr = self.fig.add_subplot(2, 1, 2)
        self.ax_tr.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        self._draw_empty()

    def _build_widgets(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Mouse Folder...", command=self._open_mouse_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self._show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.lbl_mouse = ttk.Label(top, text="Mouse: (no table loaded)"); self.lbl_mouse.pack(side=tk.LEFT, padx=(0,12))
        ttk.Label(top, text="Session:").pack(side=tk.LEFT)
        self.session_cb = ttk.Combobox(top, values=[], width=42, state="readonly")
        self.session_cb.pack(side=tk.LEFT, padx=6)
        self.session_cb.bind("<<ComboboxSelected>>", self._on_session_change)

        ttk.Checkbutton(top, text="Cells only (iscell)", variable=self.show_cells_only, command=self.refresh).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(top, text="Labels", variable=self.show_labels, command=self.refresh).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Trace:").pack(side=tk.LEFT, padx=(14,0))
        self.trace_cb = ttk.Combobox(top, values=["F (Suite2p)", "dF/F (from F)", "C (deconv calcium)"], width=18, state="readonly")
        self.trace_cb.current(0); self.trace_cb.pack(side=tk.LEFT, padx=6)
        self.trace_cb.bind("<<ComboboxSelected>>", self._on_trace_kind)

        ttk.Checkbutton(top, text="Fneu", variable=self.show_fneu, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(top, text="Spikes (S/spks)", variable=self.show_spikes, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)

        self.plot_frame = ttk.Frame(self); self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)
        bottom = ttk.Frame(self); bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status = ttk.Label(bottom, text="File → Open Mouse Folder to begin."); self.status.pack(side=tk.LEFT)

    def _show_help(self):
        messagebox.showinfo("Help",
            "Click ROI to validate across sessions.\n"
            "Popups show footprint heatmaps (not just outlines).\n"
            "If registration_results.npy is present, also shows CellReg-aligned overlay."
        )

    def _open_mouse_folder(self):
        folder = filedialog.askdirectory(title="Select mouse folder (contains mouse_table.pkl/csv)")
        if not folder:
            return
        folder = Path(folder)
        t = _load_mouse_table_from_folder(folder)
        # robust sort for session list
        if "UnixTime" in t.columns:
            t["_ut"] = pd.to_numeric(t["UnixTime"], errors="coerce")
            t = t.sort_values(["_ut","Session"], kind="mergesort").drop(columns=["_ut"], errors="ignore")
        self.t = t
        self.mouse_folder = folder
        self.fov_regs = _load_fov_regs(folder)
        self.cache.clear()

        self.sessions = list(pd.unique(t["Session"]))
        self.current_session = self.sessions[0] if self.sessions else None
        self.lbl_mouse.configure(text=f"Mouse: {str(pd.unique(t['MouseName'])[0]) if 'MouseName' in t.columns else folder.name}")
        self.session_cb.configure(values=self.sessions)
        if self.sessions:
            self.session_cb.current(0)
        self.status.configure(text="Loaded. Click a ROI outline.")
        self.refresh()

    def _on_session_change(self, _evt=None):
        if not self.sessions:
            return
        self.current_session = self.sessions[int(self.session_cb.current())]
        self.refresh()

    def _on_trace_kind(self, _evt=None):
        self.trace_kind.set(self.trace_cb.get()); self._refresh_trace_only()

    def _refresh_trace_only(self):
        self.canvas.draw_idle()

    def _draw_empty(self):
        self.ax_img.cla(); self.ax_tr.cla()
        self.ax_img.set_axis_off(); self.ax_tr.set_axis_off()
        self.ax_img.text(0.5,0.5,"File → Open Mouse Folder", ha="center", va="center", transform=self.ax_img.transAxes)
        self.canvas.draw_idle()

    def refresh(self):
        if self.t is None or self.current_session is None:
            self._draw_empty(); return
        self._draw_session()
        self.ax_tr.cla(); self.ax_tr.grid(True, alpha=0.3)
        self.ax_tr.set_xlabel("Time (s)"); self.ax_tr.set_ylabel("Signal")
        self.canvas.draw_idle()

    def _draw_session(self):
        assert self.t is not None and self.current_session is not None
        self.ax_img.cla(); self.ax_img.set_axis_off()
        rows = self.t.index[self.t["Session"]==self.current_session].to_numpy()
        if rows.size==0:
            self.ax_img.set_title("No rows", fontsize=10); return
        spath = Path(str(self.t.loc[rows[0],"SessionPath"]))
        sd = self.cache.get(self.current_session, spath)
        bg = sd.bg
        self.ax_img.imshow(bg, cmap="gray", aspect="auto")
        self.ax_img.set_title(f"{self.current_session}  (Fall.mat: {sd.fall_path})", fontsize=9)

        ids = self.t.loc[rows,"suite2pID"].astype(int).to_numpy()
        if self.show_cells_only.get():
            ok = sd.iscell
            keep = np.array([(0<=i<ok.size and bool(ok[i])) for i in ids], dtype=bool)
        else:
            keep = np.ones_like(ids, dtype=bool)

        for r_i, roi_id, k in zip(rows, ids, keep):
            if not k or roi_id<0 or roi_id>=len(sd.stat): 
                continue
            poly = _roi_polygon_from_stat(sd.stat[roi_id])
            if poly is None or poly.size==0:
                continue
            patch = Polygon(poly, closed=True, fill=False, edgecolor=(0.2,0.6,1.0,0.9), linewidth=0.8, picker=True)
            patch._cellreg_row_index = int(r_i)  # type: ignore
            self.ax_img.add_patch(patch)
            if self.show_labels.get():
                self.ax_img.text(float(np.nanmean(poly[:,0])), float(np.nanmean(poly[:,1])), str(int(roi_id)), color="yellow", fontsize=7)

        self.ax_img.set_xlim([0,bg.shape[1]]); self.ax_img.set_ylim([bg.shape[0],0])

    def _on_pick(self, event):
        artist = event.artist
        row_idx = getattr(artist, "_cellreg_row_index", None)
        if row_idx is None or self.t is None:
            return
        self._select_row(int(row_idx))

    def _select_row(self, row_idx: int):
        assert self.t is not None
        cid = int(self.t.loc[row_idx,"cellRegID"])
        sib = self.t.index[self.t["cellRegID"]==cid].to_numpy()
        sib = _safe_sort_rows_by_unix_time(self.t, sib)

        # fovID for overlay
        fov_id = None
        if "fovID" in self.t.columns:
            try:
                fov_id = int(pd.to_numeric(self.t.loc[sib,"fovID"], errors="coerce").mode().iloc[0])
            except Exception:
                fov_id = None

        self.status.configure(text=f"Selected cellRegID={cid}")
        self._popup_native(sib, cid)
        self._popup_overlay(sib, cid, fov_id)
        self._plot_traces(sib)
        self.canvas.draw_idle()

    def _get_trace_arrays(self, sd: SessionData):
        kind = self.trace_kind.get()
        base = sd.F
        name = "F"
        if kind.startswith("C"):
            base = sd.C if sd.C is not None else sd.F
            name = "C" if sd.C is not None else "F (no C)"
        elif kind.startswith("dF/F"):
            if sd.F is not None:
                base = sd.F.astype(float)
                b = np.nanpercentile(base, 10, axis=1, keepdims=True)
                b = np.where(b==0, np.nan, b)
                base = (base-b)/b
                name = "dF/F"
        fneu = sd.Fneu if self.show_fneu.get() else None
        spk = (sd.S if sd.S is not None else sd.spks) if self.show_spikes.get() else None
        fs = sd.fs if np.isfinite(sd.fs) else 1.0
        return base, fneu, spk, fs, name

    def _plot_traces(self, rows: np.ndarray):
        assert self.t is not None
        self.ax_tr.cla(); self.ax_tr.grid(True, alpha=0.3)

        if hasattr(self, "_ax_spk") and self._ax_spk is not None:
            try: self._ax_spk.remove()
            except Exception: pass
            self._ax_spk = None

        lines=[]; labels=[]; ylabel="Signal"
        for r in rows:
            sess = str(self.t.loc[r,"Session"])
            spath = Path(str(self.t.loc[r,"SessionPath"]))
            roi = int(self.t.loc[r,"suite2pID"])
            sd = self.cache.get(sess, spath)
            base, fneu, spk, fs, name = self._get_trace_arrays(sd)
            ylabel = name
            if base is None or roi<0 or roi>=base.shape[0]:
                continue
            y = np.asarray(base[roi], dtype=float)
            tt = np.arange(y.size)/fs
            ln = self.ax_tr.plot(tt, y, linewidth=1.0)[0]
            lines.append(ln); labels.append(sess)

            if fneu is not None and roi < fneu.shape[0]:
                self.ax_tr.plot(tt, np.asarray(fneu[roi], dtype=float), linewidth=0.8, alpha=0.5, linestyle="--")

            if spk is not None and roi < spk.shape[0]:
                if self._ax_spk is None:
                    self._ax_spk = self.ax_tr.twinx()
                    self._ax_spk.set_ylabel("S/spks")
                s = np.asarray(spk[roi], dtype=float)
                idx = np.where(s!=0)[0]
                if idx.size:
                    self._ax_spk.vlines(idx/fs, 0, s[idx], alpha=0.25)

        self.ax_tr.set_xlabel("Time (s)"); self.ax_tr.set_ylabel(ylabel)
        if labels:
            self.ax_tr.legend(lines, labels, fontsize=7, frameon=False, loc="upper right")

    def _popup_native(self, rows: np.ndarray, cid: int):
        assert self.t is not None
        n=len(rows); cols=int(np.ceil(np.sqrt(n))); nrows=int(np.ceil(n/cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(min(16, 4*cols), min(12, 3.5*nrows)))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes: ax.axis("off")
        cmap = plt.cm.get_cmap("tab10", max(1,n))

        for j,r in enumerate(rows):
            ax=axes[j]
            sess=str(self.t.loc[r,"Session"])
            spath=Path(str(self.t.loc[r,"SessionPath"]))
            roi=int(self.t.loc[r,"suite2pID"])
            sd=self.cache.get(sess, spath)
            bg=sd.bg
            ax.imshow(bg, cmap="gray"); ax.set_title(sess, fontsize=8)

            if 0<=roi<len(sd.stat):
                fp=_roi_footprint_image_from_stat(sd.stat[roi], bg.shape[:2])
                if fp is not None and np.nanmax(fp)>0:
                    ax.imshow(fp/float(np.nanmax(fp)), cmap="viridis", alpha=0.7)
                poly=_roi_polygon_from_stat(sd.stat[roi])
                if poly is not None and poly.size:
                    ax.plot(poly[:,0], poly[:,1], color=cmap(j), linewidth=1.6)

        fig.suptitle(f"Spatial footprints across sessions | CellRegID {cid}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0,0,1,0.95])
        plt.show(block=False)

    def _popup_overlay(self, rows: np.ndarray, cid: int, fov_id: Optional[int]):
        if fov_id is None or fov_id not in self.fov_regs or sktransform is None:
            return
        assert self.t is not None
        reg=self.fov_regs[fov_id]
        mia=reg.get("mean_image_alignment",{})
        trans=mia.get("translations", None)
        ref_idx=int(mia.get("reference_session_index",0))
        sess_paths=reg.get("session_paths",[])
        try:
            trans=np.asarray(trans, dtype=float)
        except Exception:
            return
        if trans.ndim!=2 or trans.shape[0]<3:
            return

        label_to_col={}
        for i,p in enumerate(sess_paths):
            lab=_guess_session_label_from_path(str(p), self.sessions)
            if lab not in label_to_col: label_to_col[lab]=i
        ref_label=None
        if 0<=ref_idx<len(sess_paths):
            ref_label=_guess_session_label_from_path(str(sess_paths[ref_idx]), self.sessions)

        # choose reference bg from the sibling that matches ref_label if possible
        ref_row=rows[0]
        if ref_label is not None:
            cand=[r for r in rows if str(self.t.loc[r,"Session"])==ref_label]
            if cand: ref_row=cand[0]
        ref_sess=str(self.t.loc[ref_row,"Session"])
        ref_path=Path(str(self.t.loc[ref_row,"SessionPath"]))
        ref_sd=self.cache.get(ref_sess, ref_path)
        bg=ref_sd.bg; H,W=bg.shape[:2]

        fig, ax = plt.subplots(1,1, figsize=(7.5,7.5))
        ax.imshow(bg, cmap="gray"); ax.axis("off")
        ax.set_title(f"CellReg-aligned overlay (FOV{fov_id}) | CellRegID {cid}", fontsize=10)
        cmap = plt.cm.get_cmap("tab10", max(1,len(rows)))
        handles=[]

        for j,r in enumerate(rows):
            sess=str(self.t.loc[r,"Session"])
            spath=Path(str(self.t.loc[r,"SessionPath"]))
            roi=int(self.t.loc[r,"suite2pID"])
            sd=self.cache.get(sess, spath)
            if not (0<=roi<len(sd.stat)): 
                continue
            fp=_roi_footprint_image_from_stat(sd.stat[roi], (H,W))
            if fp is None or np.nanmax(fp)<=0:
                continue
            col=label_to_col.get(sess, None)
            if col is None or col>=trans.shape[1]:
                continue
            dx,dy,rot=float(trans[0,col]), float(trans[1,col]), float(trans[2,col])
            tform=sktransform.EuclideanTransform(translation=(dx,dy), rotation=np.deg2rad(rot))
            fpw=sktransform.warp(fp, tform.inverse, output_shape=(H,W), order=1, preserve_range=True, mode="constant", cval=0.0)
            thr=float(np.nanmax(fpw))*0.35
            if thr<=0: 
                continue
            mask=fpw>thr
            cs=ax.contour(mask.astype(float), levels=[0.5], colors=[cmap(j)], linewidths=2.0)
            if cs.collections:
                h=cs.collections[0]; h.set_label(sess); handles.append(h)

        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
        plt.show(block=False)


def launch():
    Viewer().mainloop()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mouse-folder", type=str, default=None)
    args,_=ap.parse_known_args()
    app=Viewer()
    if args.mouse_folder:
        # auto-load
        folder=Path(args.mouse_folder)
        t=_load_mouse_table_from_folder(folder)
        if "UnixTime" in t.columns:
            t["_ut"]=pd.to_numeric(t["UnixTime"], errors="coerce")
            t=t.sort_values(["_ut","Session"], kind="mergesort").drop(columns=["_ut"], errors="ignore")
        app.t=t; app.mouse_folder=folder; app.fov_regs=_load_fov_regs(folder); app.cache.clear()
        app.sessions=list(pd.unique(t["Session"])); app.current_session=app.sessions[0] if app.sessions else None
        app.lbl_mouse.configure(text=f"Mouse: {folder.name}")
        app.session_cb.configure(values=app.sessions)
        if app.sessions: app.session_cb.current(0)
        app.status.configure(text="Loaded. Click a ROI outline.")
        app.refresh()
    app.mainloop()


if __name__=="__main__":
    main()

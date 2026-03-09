
'''
viz_cellreg_alignment.py

Python GUI to validate CellRegPy registration using mouse_table outputs.

MATLAB reference: vizCellRegAlignment(mouse_table, mouse_data)
- Session dropdown + mode dropdown
- Image axis showing background + ROI outlines; clicking selects ROI
- Shows footprints across sessions + plots traces (F / C / S)

This script provides the analogous validation workflow in Python.

Expected inputs:
- mouse_table (csv/pkl) produced by CellRegPy run_pipeline
  columns: MouseName, Session, UnixTime, suite2pID, cellRegID, fovID, SessionPath
- Suite2p outputs reachable from SessionPath (expects suite2p/plane0/Fall.mat somewhere under SessionPath)

Usage:
  python viz_cellreg_alignment.py --mouse-folder \"C:\\path\\to\\mouse_folder\"
  python viz_cellreg_alignment.py --mouse-table  \"C:\\path\\to\\mouse_table.pkl\"
'''

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

# GUI
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use(\"TkAgg\")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None  # fallback to raw pixel cloud

# Try to reuse loaders from your package if available
try:
    from cellregpy.cellregpy import load_fall_mat  # type: ignore
except Exception:
    load_fall_mat = None


def _truncate(s: str, n: int = 22) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n]


def _find_fall_mat(session_path: Path) -> Path:
    \"\"\"Find suite2p/plane0/Fall.mat under a session path.\"\"\"
    session_path = Path(session_path)
    candidates = list(session_path.rglob(\"suite2p/plane0/Fall.mat\"))
    if not candidates:
        if (session_path / \"Fall.mat\").exists():
            return session_path / \"Fall.mat\"
        raise FileNotFoundError(f\"Could not find suite2p/plane0/Fall.mat under: {session_path}\")
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def _unwrap_mat_struct(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.size == 1:
        try:
            return x.item()
        except Exception:
            return x
    return x


@dataclass
class SessionData:
    session: str
    session_path: Path
    fall_path: Path
    ops: Dict[str, Any]
    stat: List[Any]
    iscell: np.ndarray
    mean_img: np.ndarray
    max_proj: np.ndarray
    fs: float
    F: Optional[np.ndarray] = None
    Fneu: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    spks: Optional[np.ndarray] = None


class SessionCache:
    def __init__(self):
        self._cache: Dict[str, SessionData] = {}

    def get(self, session: str, session_path: Path) -> SessionData:
        key = f\"{session}::{session_path}\"
        if key in self._cache:
            return self._cache[key]
        sd = self._load_session(session, session_path)
        self._cache[key] = sd
        return sd

    def _load_session(self, session: str, session_path: Path) -> SessionData:
        fall_path = _find_fall_mat(session_path)
        if load_fall_mat is not None:
            fall = load_fall_mat(fall_path.parent)  # plane0 folder
        else:
            from scipy.io import loadmat
            fall = loadmat(fall_path, simplify_cells=False)

        ops = _unwrap_mat_struct(fall.get(\"ops\", None))
        stat = _unwrap_mat_struct(fall.get(\"stat\", None))
        iscell = _unwrap_mat_struct(fall.get(\"iscell\", None))

        # ops -> dict
        if isinstance(ops, dict):
            ops_d = ops
        else:
            ops_d = {}
            try:
                for name in ops.dtype.names:
                    ops_d[name] = _unwrap_mat_struct(ops[name])
            except Exception:
                ops_d = {}

        # stat -> list
        stat_list: List[Any] = []
        if isinstance(stat, (list, tuple)):
            stat_list = list(stat)
        elif isinstance(stat, np.ndarray):
            if stat.dtype == object:
                stat_list = list(stat.flat)
            else:
                stat_list = [stat[i] for i in range(stat.shape[0])]

        # iscell -> bool vector
        if isinstance(iscell, np.ndarray):
            if iscell.ndim == 2:
                iscell_b = iscell[:, 0].astype(bool)
            else:
                iscell_b = iscell.astype(bool)
        else:
            iscell_b = np.ones(len(stat_list), dtype=bool)

        # images
        mean_img = None
        for k in (\"meanImgE\", \"meanImg\", \"max_proj\"):
            if k in ops_d and ops_d[k] is not None:
                try:
                    mean_img = np.asarray(ops_d[k], dtype=float)
                    break
                except Exception:
                    pass
        if mean_img is None:
            mean_img = np.zeros((512, 512), dtype=float)

        max_proj = mean_img
        if \"max_proj\" in ops_d and ops_d[\"max_proj\"] is not None:
            try:
                max_proj = np.asarray(ops_d[\"max_proj\"], dtype=float)
            except Exception:
                max_proj = mean_img

        fs = float(ops_d.get(\"fs\", np.nan))
        if not np.isfinite(fs) or fs <= 0:
            fs = np.nan

        def grab(name: str) -> Optional[np.ndarray]:
            v = fall.get(name, None)
            if v is None:
                return None
            try:
                v = _unwrap_mat_struct(v)
                return np.asarray(v)
            except Exception:
                return None

        F = grab(\"F\")
        Fneu = grab(\"Fneu\")
        C = grab(\"C\")
        S = grab(\"S\")
        spks = grab(\"spks\") or grab(\"s2pSpk\")

        return SessionData(
            session=session,
            session_path=Path(session_path),
            fall_path=fall_path,
            ops=ops_d,
            stat=stat_list,
            iscell=iscell_b,
            mean_img=mean_img,
            max_proj=max_proj,
            fs=fs,
            F=F,
            Fneu=Fneu,
            C=C,
            S=S,
            spks=spks,
        )


def _roi_polygon_from_stat(stat_i: Any) -> Optional[np.ndarray]:
    try:
        if isinstance(stat_i, dict):
            x = np.asarray(stat_i.get(\"xpix\"), dtype=float).ravel()
            y = np.asarray(stat_i.get(\"ypix\"), dtype=float).ravel()
        else:
            x = np.asarray(stat_i[\"xpix\"], dtype=float).ravel()
            y = np.asarray(stat_i[\"ypix\"], dtype=float).ravel()

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


class CellRegViewer(tk.Tk):
    def __init__(self, mouse_table: pd.DataFrame):
        super().__init__()
        self.title(\"CellRegPy Viewer (validation)\")
        self.geometry(\"1200x780\")

        self.t = mouse_table.copy()
        if \"UnixTime\" in self.t.columns:
            self.t = self.t.sort_values(\"UnixTime\", kind=\"mergesort\")

        self.sessions = list(pd.unique(self.t[\"Session\"]))
        self.mouse_name = str(pd.unique(self.t.get(\"MouseName\", [\"mouse\"]))[0]) if \"MouseName\" in self.t.columns else \"mouse\"

        self.cache = SessionCache()

        self.current_session = self.sessions[0]
        self.show_cells_only = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)

        self.trace_kind = tk.StringVar(value=\"F\")  # F / dFF / C
        self.show_fneu = tk.BooleanVar(value=False)
        self.show_spikes = tk.BooleanVar(value=True)

        self._build_widgets()

        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_img = self.fig.add_subplot(2, 1, 1)
        self.ax_tr = self.fig.add_subplot(2, 1, 2)
        self.ax_tr.set_xlabel(\"Time (s)\")
        self.ax_tr.set_ylabel(\"Fluorescence\")
        self.ax_tr.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect(\"pick_event\", self._on_pick)

        self.refresh()

    def _build_widgets(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text=f\"Mouse: {self.mouse_name}\").pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top, text=\"Session:\").pack(side=tk.LEFT)
        self.session_cb = ttk.Combobox(top, values=[_truncate(s, 32) for s in self.sessions], width=38, state=\"readonly\")
        self.session_cb.current(0)
        self.session_cb.pack(side=tk.LEFT, padx=6)
        self.session_cb.bind(\"<<ComboboxSelected>>\", self._on_session_change)

        ttk.Checkbutton(top, text=\"Cells only (iscell)\", variable=self.show_cells_only, command=self.refresh).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(top, text=\"Labels\", variable=self.show_labels, command=self.refresh).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text=\"Trace:\").pack(side=tk.LEFT, padx=(14, 0))
        self.trace_cb = ttk.Combobox(top, values=[\"F\", \"dFF\", \"C\"], width=6, state=\"readonly\")
        self.trace_cb.current(0)
        self.trace_cb.pack(side=tk.LEFT, padx=6)
        self.trace_cb.bind(\"<<ComboboxSelected>>\", self._on_trace_kind)

        ttk.Checkbutton(top, text=\"Fneu\", variable=self.show_fneu, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(top, text=\"Spikes (S/spks)\", variable=self.show_spikes, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)

        ttk.Button(top, text=\"Help\", command=self._show_help).pack(side=tk.RIGHT)

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status = ttk.Label(bottom, text=\"Click a ROI outline to inspect across sessions.\")
        self.status.pack(side=tk.LEFT)

    def _show_help(self):
        messagebox.showinfo(
            \"Help\",
            \"Pick a session → click a ROI outline.\\n\"
            \"A separate window shows that CellRegID across sessions.\\n\"
            \"Bottom panel shows traces (F/C/dF/F) + optional spikes.\"
        )

    def _on_session_change(self, _evt=None):
        idx = int(self.session_cb.current())
        self.current_session = self.sessions[idx]
        self.refresh()

    def _on_trace_kind(self, _evt=None):
        self.trace_kind.set(self.trace_cb.get())
        self._refresh_trace_only()

    def _refresh_trace_only(self):
        # traces update only after selection; no selection -> noop
        self.canvas.draw_idle()

    def refresh(self):
        self._draw_session_image_and_rois()
        self.ax_tr.cla()
        self.ax_tr.set_xlabel(\"Time (s)\")
        self.ax_tr.set_ylabel(\"Fluorescence\")
        self.ax_tr.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _draw_session_image_and_rois(self):
        self.ax_img.cla()
        self.ax_img.set_axis_off()

        rows = self.t.index[self.t[\"Session\"] == self.current_session].to_numpy()
        if rows.size == 0:
            self.ax_img.set_title(\"No rows for session\", fontsize=10)
            return

        session_path = Path(str(self.t.loc[rows[0], \"SessionPath\"]))
        sd = self.cache.get(self.current_session, session_path)

        bg = np.asarray(sd.max_proj if sd.max_proj is not None else sd.mean_img, dtype=float)
        self.ax_img.imshow(bg, cmap=\"gray\")
        self.ax_img.set_title(_truncate(self.current_session, 50), fontsize=10)

        ids = self.t.loc[rows, \"suite2pID\"].astype(int).to_numpy()

        # filter by iscell if requested
        draw_mask = np.ones(ids.shape[0], dtype=bool)
        if self.show_cells_only.get():
            ok = sd.iscell
            draw_mask = np.array([(0 <= i < ok.size and bool(ok[i])) for i in ids], dtype=bool)

        for j, (r_i, roi_id) in enumerate(zip(rows, ids)):
            if not draw_mask[j]:
                continue
            if roi_id < 0 or roi_id >= len(sd.stat):
                continue
            poly = _roi_polygon_from_stat(sd.stat[roi_id])
            if poly is None or poly.size == 0:
                continue

            patch = Polygon(poly, closed=True, fill=False, edgecolor=(0.2, 0.6, 1.0, 0.9), linewidth=0.8, picker=True)
            patch._cellreg_row_index = int(r_i)  # type: ignore[attr-defined]
            self.ax_img.add_patch(patch)

            if self.show_labels.get():
                x0 = float(np.nanmean(poly[:, 0]))
                y0 = float(np.nanmean(poly[:, 1]))
                self.ax_img.text(x0, y0, str(int(roi_id)), color=\"yellow\", fontsize=7)

        self.ax_img.set_xlim([0, bg.shape[1]])
        self.ax_img.set_ylim([bg.shape[0], 0])

    def _on_pick(self, event):
        artist = event.artist
        row_idx = getattr(artist, \"_cellreg_row_index\", None)
        if row_idx is None:
            return
        self._select_row(int(row_idx))

    def _select_row(self, row_idx: int):
        if row_idx not in self.t.index:
            return
        cid = int(self.t.loc[row_idx, \"cellRegID\"])
        self.status.configure(text=f\"Selected cellRegID={cid} from session={self.t.loc[row_idx, 'Session']}\")
        siblings = self.t.index[self.t[\"cellRegID\"] == cid].to_numpy()
        if siblings.size == 0:
            return
        if \"UnixTime\" in self.t.columns:
            siblings = siblings[np.argsort(self.t.loc[siblings, \"UnixTime\"].to_numpy())]

        self._highlight_selected(row_idx)
        self._plot_traces_for_rows(siblings)
        self._show_footprints_window(siblings, cid)
        self.canvas.draw_idle()

    def _highlight_selected(self, row_idx: int):
        self._draw_session_image_and_rois()
        roi_id = int(self.t.loc[row_idx, \"suite2pID\"])
        sess = str(self.t.loc[row_idx, \"Session\"])
        sess_path = Path(str(self.t.loc[row_idx, \"SessionPath\"]))
        sd = self.cache.get(sess, sess_path)

        if 0 <= roi_id < len(sd.stat):
            poly = _roi_polygon_from_stat(sd.stat[roi_id])
            if poly is not None and poly.size:
                patch = Polygon(poly, closed=True, fill=True, facecolor=(1.0, 0.2, 0.2, 0.25), edgecolor=\"red\", linewidth=1.5)
                self.ax_img.add_patch(patch)

    def _get_trace_arrays(self, sd: SessionData) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, str]:
        fs = sd.fs
        kind = self.trace_kind.get()
        name = \"F\"
        base = None

        if kind == \"C\" and sd.C is not None:
            base = sd.C
            name = \"C\"
        elif sd.F is not None:
            base = sd.F
            name = \"F\"

        if kind == \"dFF\" and base is not None:
            b = np.nanpercentile(base, 10, axis=1, keepdims=True)
            b = np.where(b == 0, np.nan, b)
            base = (base - b) / b
            name = \"dF/F\"

        fneu = sd.Fneu if self.show_fneu.get() else None

        spk = None
        if self.show_spikes.get():
            spk = sd.S if sd.S is not None else sd.spks

        return base, fneu, spk, fs, name

    def _plot_traces_for_rows(self, rows: np.ndarray):
        self.ax_tr.cla()
        self.ax_tr.grid(True, alpha=0.3)

        lines = []
        labels = []

        # clear any prior twin axis
        if hasattr(self, \"_ax_spk\") and self._ax_spk is not None:
            try:
                self._ax_spk.remove()
            except Exception:
                pass
            self._ax_spk = None

        for r in rows:
            sess = str(self.t.loc[r, \"Session\"])
            sess_path = Path(str(self.t.loc[r, \"SessionPath\"]))
            roi_id = int(self.t.loc[r, \"suite2pID\"])
            sd = self.cache.get(sess, sess_path)

            base, fneu, spk, fs, name = self._get_trace_arrays(sd)
            if base is None or roi_id < 0 or roi_id >= base.shape[0]:
                continue
            if not np.isfinite(fs):
                fs = 1.0

            y = np.asarray(base[roi_id], dtype=float)
            t = np.arange(y.size) / fs
            ln = self.ax_tr.plot(t, y, linewidth=1.0)[0]
            lines.append(ln)
            labels.append(_truncate(sess, 22))

            if fneu is not None and roi_id < fneu.shape[0]:
                self.ax_tr.plot(t, np.asarray(fneu[roi_id], dtype=float), linewidth=0.8, alpha=0.5, linestyle=\"--\")

            if spk is not None and roi_id < spk.shape[0]:
                if self._ax_spk is None:
                    self._ax_spk = self.ax_tr.twinx()
                    self._ax_spk.set_ylabel(\"S/spks\")
                s = np.asarray(spk[roi_id], dtype=float)
                idx = np.where(s != 0)[0]
                if idx.size:
                    tt = idx / fs
                    self._ax_spk.vlines(tt, 0, s[idx], alpha=0.25)
                try:
                    self._ax_spk.set_ylim(0, max(1.0, float(np.nanmax(s))))
                except Exception:
                    self._ax_spk.set_ylim(0, 1.0)

        self.ax_tr.set_xlabel(\"Time (s)\")
        self.ax_tr.set_ylabel(self.trace_kind.get())
        if labels:
            self.ax_tr.legend(lines, labels, loc=\"upper right\", fontsize=7, frameon=False)

    def _show_footprints_window(self, rows: np.ndarray, cid: int):
        n = len(rows)
        cols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(nrows, cols, figsize=(min(14, 4*cols), min(10, 3*nrows)))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis(\"off\")

        cmap = plt.cm.get_cmap(\"tab10\", n)

        for j, r in enumerate(rows):
            ax = axes[j]
            sess = str(self.t.loc[r, \"Session\"])
            sess_path = Path(str(self.t.loc[r, \"SessionPath\"]))
            roi_id = int(self.t.loc[r, \"suite2pID\"])
            sd = self.cache.get(sess, sess_path)

            bg = np.asarray(sd.max_proj if sd.max_proj is not None else sd.mean_img, dtype=float)
            ax.imshow(bg, cmap=\"gray\")
            ax.set_title(_truncate(sess, 22), fontsize=8)

            if 0 <= roi_id < len(sd.stat):
                poly = _roi_polygon_from_stat(sd.stat[roi_id])
                if poly is not None and poly.size:
                    ax.plot(poly[:, 0], poly[:, 1], color=cmap(j), linewidth=1.4)

        fig.suptitle(f\"CellRegID {cid} — ROI outlines across sessions\", fontsize=12, fontweight=\"bold\")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)


def _load_mouse_table(mouse_folder: Optional[Path], mouse_table_path: Optional[Path]) -> pd.DataFrame:
    if mouse_table_path is None:
        if mouse_folder is None:
            raise ValueError(\"Provide --mouse-folder or --mouse-table\")
        mouse_folder = Path(mouse_folder)
        pkl = mouse_folder / \"mouse_table.pkl\"
        if pkl.exists():
            mouse_table_path = pkl
        else:
            csv = mouse_folder / \"mouse_table.csv\"
            if csv.exists():
                mouse_table_path = csv
            else:
                raise FileNotFoundError(\"Could not find mouse_table.pkl or mouse_table.csv in mouse folder\")

    mouse_table_path = Path(mouse_table_path)
    if mouse_table_path.suffix.lower() in (\".pkl\", \".pickle\"):
        t = pd.read_pickle(mouse_table_path)
    elif mouse_table_path.suffix.lower() == \".csv\":
        t = pd.read_csv(mouse_table_path)
    else:
        raise ValueError(f\"Unsupported mouse_table file type: {mouse_table_path}\")

    required = {\"Session\", \"suite2pID\", \"cellRegID\", \"SessionPath\"}
    missing = required - set(t.columns)
    if missing:
        raise ValueError(f\"mouse_table is missing columns: {sorted(missing)}\")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--mouse-folder\", type=str, default=None, help=\"Mouse folder containing mouse_table.pkl/csv.\")
    ap.add_argument(\"--mouse-table\", type=str, default=None, help=\"Path to mouse_table.pkl or mouse_table.csv\")
    args = ap.parse_args()

    mouse_folder = Path(args.mouse_folder) if args.mouse_folder else None
    mouse_table_path = Path(args.mouse_table) if args.mouse_table else None

    t = _load_mouse_table(mouse_folder, mouse_table_path)
    if \"MouseName\" in t.columns and t[\"MouseName\"].nunique() > 1:
        warnings.warn(\"mouse_table contains multiple MouseName values; viewer assumes a single mouse for best UX.\")

    app = CellRegViewer(t)
    app.mainloop()


if __name__ == \"__main__\":
    main()

\
"""
viz_cellreg_alignment_gui_v3.py

CellRegPy validation GUI (Tkinter + Matplotlib).

New in v3:
- Built-in "File -> Open Mouse Folder..." (and "Open Mouse Table...") so you can load data
  from inside the GUI (no CLI args required).
- If launched without arguments, it starts empty and prompts you to open a mouse folder/table.

What it validates
- Uses mouse_table.pkl/csv to find SessionPath + suite2pID + cellRegID.
- Displays a session background image (ops.max_proj or meanImgE/meanImg) + ROI outlines.
- Click ROI -> show that CellRegID across sessions (side-by-side) + traces (F/dF/F/C) and S/spks.

Notes
- "C" here refers to an optional calcium trace array stored in Fall.mat, NOT C/C++.
- Run from terminal is best, but Interactive/Notebook works if you call launch().

Usage (terminal):
  python viz_cellreg_alignment_gui_v3.py --mouse-folder "C:\\path\\to\\mouse_folder"

Usage (Interactive/Notebook):
  from viz_cellreg_alignment_gui_v3 import launch
  launch()  # prompts for folder
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

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


def _truncate(s: str, n: int = 24) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n]


def _find_fall_mat(session_path: Path) -> Path:
    session_path = Path(session_path)
    candidates = list(session_path.rglob("suite2p/plane0/Fall.mat"))
    if not candidates:
        if (session_path / "Fall.mat").exists():
            return session_path / "Fall.mat"
        raise FileNotFoundError(f"Could not find suite2p/plane0/Fall.mat under: {session_path}")
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def _as_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        return np.asarray(x)
    except Exception:
        return None


def _ops_to_dict(ops: Any) -> Dict[str, Any]:
    if ops is None:
        return {}
    if isinstance(ops, dict):
        return ops
    out: Dict[str, Any] = {}
    for k in ("meanImgE", "meanImg", "max_proj", "Ly", "Lx", "fs", "nframes"):
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
    C: Optional[np.ndarray] = None   # calcium "C" trace (not C/C++)
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
        sd = self._load_session(session, session_path)
        self._cache[key] = sd
        return sd

    def _load_session(self, session: str, session_path: Path) -> SessionData:
        fall_path = _find_fall_mat(session_path)
        data = loadmat(str(fall_path), squeeze_me=True, struct_as_record=False)

        ops = _ops_to_dict(data.get("ops", None))
        stat_list = _stat_to_list(data.get("stat", None))
        n_rois = len(stat_list)
        iscell_b = _iscell_to_boolvec(data.get("iscell", None), n_rois)

        mean_img = None
        for k in ("max_proj", "meanImgE", "meanImg"):
            if k in ops and ops[k] is not None:
                try:
                    mean_img = np.asarray(ops[k], dtype=float)
                    break
                except Exception:
                    pass
        if mean_img is None:
            mean_img = np.zeros((512, 512), dtype=float)

        max_proj = mean_img
        if "max_proj" in ops and ops["max_proj"] is not None:
            try:
                max_proj = np.asarray(ops["max_proj"], dtype=float)
            except Exception:
                max_proj = mean_img

        fs = float(ops.get("fs", np.nan))
        if not np.isfinite(fs) or fs <= 0:
            fs = np.nan

        F = _as_array(data.get("F", None))
        Fneu = _as_array(data.get("Fneu", None))
        C = _as_array(data.get("C", None))
        S = _as_array(data.get("S", None))
        spks = _as_array(data.get("spks", None))
        if spks is None:
            spks = _as_array(data.get("s2pSpk", None))

        return SessionData(
            session=session,
            session_path=Path(session_path),
            fall_path=fall_path,
            ops=ops,
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


class CellRegViewer(tk.Tk):
    def __init__(self, mouse_table: Optional[pd.DataFrame] = None):
        super().__init__()
        self.title("CellRegPy Viewer (validation)")
        self.geometry("1200x780")

        self.cache = SessionCache()

        self.t: Optional[pd.DataFrame] = None
        self.sessions: List[str] = []
        self.mouse_name: str = "mouse"
        self.current_session: Optional[str] = None

        self.show_cells_only = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.trace_kind = tk.StringVar(value="F (Suite2p)")
        self.show_fneu = tk.BooleanVar(value=False)
        self.show_spikes = tk.BooleanVar(value=True)

        self._build_widgets()

        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_img = self.fig.add_subplot(2, 1, 1)
        self.ax_tr = self.fig.add_subplot(2, 1, 2)
        self.ax_tr.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        if mouse_table is not None:
            self.set_table(mouse_table)
        else:
            self._draw_empty()

    def _build_widgets(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Mouse Folder...", command=self._open_mouse_folder)
        file_menu.add_command(label="Open Mouse Table...", command=self._open_mouse_table)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Help", command=self._show_help)
        menubar.add_cascade(label="Help", menu=help_menu)

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
        ttk.Checkbutton(top, text="Labels", variable=self.show_labels, command=self.refresh).pack(side=tk.LEFT, padx=10)

        ttk.Label(top, text="Trace:").pack(side=tk.LEFT, padx=(14, 0))
        self.trace_cb = ttk.Combobox(
            top,
            values=["F (Suite2p)", "dF/F (from F)", "C (deconv calcium)"],
            width=18,
            state="readonly"
        )
        self.trace_cb.current(0)
        self.trace_cb.pack(side=tk.LEFT, padx=6)
        self.trace_cb.bind("<<ComboboxSelected>>", self._on_trace_kind)

        ttk.Checkbutton(top, text="Fneu", variable=self.show_fneu, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(top, text="Spikes (S/spks)", variable=self.show_spikes, command=self._refresh_trace_only).pack(side=tk.LEFT, padx=8)

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status = ttk.Label(bottom, text="File → Open Mouse Folder/Table to begin.")
        self.status.pack(side=tk.LEFT)

    def _show_help(self):
        msg = """Workflow:
1) File → Open Mouse Folder... (or Open Mouse Table...)
2) Pick a session in the dropdown.
3) Click a ROI outline.
4) A window pops up showing that CellRegID across sessions.
5) Bottom panel shows traces.

Trace naming:
- F (Suite2p): raw fluorescence (Fall.mat: F)
- dF/F (from F): computed from F (10th percentile baseline)
- C (deconv calcium): optional Fall.mat variable 'C' (if missing, viewer falls back to F and labels it)
- Spikes: uses 'S' if present else 'spks'/'s2pSpk'
"""
        messagebox.showinfo("Help", msg)

    def _open_mouse_folder(self):
        folder = filedialog.askdirectory(title="Select mouse folder (contains mouse_table.pkl/csv)")
        if not folder:
            return
        try:
            t = _load_mouse_table(mouse_folder=Path(folder))
            self.set_table(t)
        except Exception as e:
            messagebox.showerror("Failed to load mouse folder", str(e))

    def _open_mouse_table(self):
        f = filedialog.askopenfilename(
            title="Select mouse_table.pkl or mouse_table.csv",
            filetypes=[("Mouse table", "*.pkl *.pickle *.csv"), ("All files", "*.*")]
        )
        if not f:
            return
        try:
            t = _load_mouse_table(mouse_table_path=Path(f))
            self.set_table(t)
        except Exception as e:
            messagebox.showerror("Failed to load mouse table", str(e))

    def set_table(self, mouse_table: pd.DataFrame):
        self.cache.clear()

        t = mouse_table.copy()
        if "UnixTime" in t.columns:
            t = t.sort_values("UnixTime", kind="mergesort")

        self.t = t
        self.sessions = list(pd.unique(t["Session"]))
        self.mouse_name = str(pd.unique(t["MouseName"])[0]) if "MouseName" in t.columns else "mouse"
        self.current_session = self.sessions[0] if self.sessions else None

        self.lbl_mouse.configure(text=f"Mouse: {self.mouse_name}")
        self.session_cb.configure(values=[_truncate(s, 40) for s in self.sessions])
        if self.sessions:
            self.session_cb.current(0)

        self.status.configure(text="Loaded mouse_table. Click a ROI outline to inspect across sessions.")
        self.refresh()

    def _draw_empty(self):
        self.ax_img.cla()
        self.ax_tr.cla()
        self.ax_img.set_axis_off()
        self.ax_tr.set_axis_off()
        self.ax_img.text(0.5, 0.5, "File → Open Mouse Folder/Table", ha="center", va="center", transform=self.ax_img.transAxes)
        self.canvas.draw_idle()

    def _on_session_change(self, _evt=None):
        if not self.sessions:
            return
        idx = int(self.session_cb.current())
        self.current_session = self.sessions[idx]
        self.refresh()

    def _on_trace_kind(self, _evt=None):
        self.trace_kind.set(self.trace_cb.get())
        self._refresh_trace_only()

    def _refresh_trace_only(self):
        self.canvas.draw_idle()

    def refresh(self):
        if self.t is None or not self.sessions or self.current_session is None:
            self._draw_empty()
            return

        self._draw_session_image_and_rois()
        self.ax_tr.cla()
        self.ax_tr.grid(True, alpha=0.3)
        self.ax_tr.set_xlabel("Time (s)")
        self.ax_tr.set_ylabel("Signal")
        self.canvas.draw_idle()

    def _draw_session_image_and_rois(self):
        assert self.t is not None
        assert self.current_session is not None

        self.ax_img.cla()
        self.ax_img.set_axis_off()

        rows = self.t.index[self.t["Session"] == self.current_session].to_numpy()
        if rows.size == 0:
            self.ax_img.set_title("No rows for session", fontsize=10)
            return

        session_path = Path(str(self.t.loc[rows[0], "SessionPath"]))
        sd = self.cache.get(self.current_session, session_path)

        bg = np.asarray(sd.max_proj if sd.max_proj is not None else sd.mean_img, dtype=float)
        self.ax_img.imshow(bg, cmap="gray")
        self.ax_img.set_title(_truncate(self.current_session, 60), fontsize=10)

        ids = self.t.loc[rows, "suite2pID"].astype(int).to_numpy()

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
                self.ax_img.text(x0, y0, str(int(roi_id)), color="yellow", fontsize=7)

        self.ax_img.set_xlim([0, bg.shape[1]])
        self.ax_img.set_ylim([bg.shape[0], 0])

    def _on_pick(self, event):
        if self.t is None:
            return
        artist = event.artist
        row_idx = getattr(artist, "_cellreg_row_index", None)
        if row_idx is None:
            return
        self._select_row(int(row_idx))

    def _select_row(self, row_idx: int):
        assert self.t is not None

        if row_idx not in self.t.index:
            return

        cid = int(self.t.loc[row_idx, "cellRegID"])
        self.status.configure(text=f"Selected cellRegID={cid} from session={self.t.loc[row_idx, 'Session']}")

        siblings = self.t.index[self.t["cellRegID"] == cid].to_numpy()
        if siblings.size == 0:
            return
        if "UnixTime" in self.t.columns:
            siblings = siblings[np.argsort(self.t.loc[siblings, "UnixTime"].to_numpy())]

        self._highlight_selected(row_idx)
        self._plot_traces_for_rows(siblings)
        self._show_footprints_window(siblings, cid)
        self.canvas.draw_idle()

    def _highlight_selected(self, row_idx: int):
        self._draw_session_image_and_rois()
        assert self.t is not None

        roi_id = int(self.t.loc[row_idx, "suite2pID"])
        sess = str(self.t.loc[row_idx, "Session"])
        sess_path = Path(str(self.t.loc[row_idx, "SessionPath"]))
        sd = self.cache.get(sess, sess_path)

        if 0 <= roi_id < len(sd.stat):
            poly = _roi_polygon_from_stat(sd.stat[roi_id])
            if poly is not None and poly.size:
                patch = Polygon(poly, closed=True, fill=True, facecolor=(1.0, 0.2, 0.2, 0.25), edgecolor="red", linewidth=1.5)
                self.ax_img.add_patch(patch)

    def _get_trace_arrays(self, sd: SessionData) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, str]:
        fs = sd.fs
        kind = self.trace_kind.get()

        base = None
        name = "F"

        if kind.startswith("C"):
            if sd.C is not None:
                base = sd.C
                name = "C (deconv)"
            elif sd.F is not None:
                base = sd.F
                name = "F (no C in Fall.mat)"
        elif kind.startswith("dF/F"):
            if sd.F is not None:
                base = sd.F.astype(float)
                b = np.nanpercentile(base, 10, axis=1, keepdims=True)
                b = np.where(b == 0, np.nan, b)
                base = (base - b) / b
                name = "dF/F (from F)"
        else:
            base = sd.F
            name = "F"

        fneu = sd.Fneu if self.show_fneu.get() else None
        spk = None
        if self.show_spikes.get():
            spk = sd.S if sd.S is not None else sd.spks

        return base, fneu, spk, fs, name

    def _plot_traces_for_rows(self, rows: np.ndarray):
        assert self.t is not None

        self.ax_tr.cla()
        self.ax_tr.grid(True, alpha=0.3)

        if hasattr(self, "_ax_spk") and self._ax_spk is not None:
            try:
                self._ax_spk.remove()
            except Exception:
                pass
            self._ax_spk = None

        lines = []
        labels = []
        ylabel = "Signal"

        for r in rows:
            sess = str(self.t.loc[r, "Session"])
            sess_path = Path(str(self.t.loc[r, "SessionPath"]))
            roi_id = int(self.t.loc[r, "suite2pID"])
            sd = self.cache.get(sess, sess_path)

            base, fneu, spk, fs, name = self._get_trace_arrays(sd)
            ylabel = name

            if base is None or roi_id < 0 or roi_id >= base.shape[0]:
                continue
            if not np.isfinite(fs):
                fs = 1.0

            y = np.asarray(base[roi_id], dtype=float)
            tt = np.arange(y.size) / fs
            ln = self.ax_tr.plot(tt, y, linewidth=1.0)[0]
            lines.append(ln)
            labels.append(_truncate(sess, 28))

            if fneu is not None and roi_id < fneu.shape[0]:
                self.ax_tr.plot(tt, np.asarray(fneu[roi_id], dtype=float), linewidth=0.8, alpha=0.5, linestyle="--")

            if spk is not None and roi_id < spk.shape[0]:
                if self._ax_spk is None:
                    self._ax_spk = self.ax_tr.twinx()
                    self._ax_spk.set_ylabel("S/spks")
                s = np.asarray(spk[roi_id], dtype=float)
                idx = np.where(s != 0)[0]
                if idx.size:
                    self._ax_spk.vlines(idx / fs, 0, s[idx], alpha=0.25)
                try:
                    self._ax_spk.set_ylim(0, max(1.0, float(np.nanmax(s))))
                except Exception:
                    self._ax_spk.set_ylim(0, 1.0)

        self.ax_tr.set_xlabel("Time (s)")
        self.ax_tr.set_ylabel(ylabel)
        if labels:
            self.ax_tr.legend(lines, labels, loc="upper right", fontsize=7, frameon=False)

    def _show_footprints_window(self, rows: np.ndarray, cid: int):
        assert self.t is not None

        n = len(rows)
        cols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(nrows, cols, figsize=(min(14, 4 * cols), min(10, 3 * nrows)))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")

        cmap = plt.cm.get_cmap("tab10", max(1, n))

        for j, r in enumerate(rows):
            ax = axes[j]
            sess = str(self.t.loc[r, "Session"])
            sess_path = Path(str(self.t.loc[r, "SessionPath"]))
            roi_id = int(self.t.loc[r, "suite2pID"])
            sd = self.cache.get(sess, sess_path)

            bg = np.asarray(sd.max_proj if sd.max_proj is not None else sd.mean_img, dtype=float)
            ax.imshow(bg, cmap="gray")
            ax.set_title(_truncate(sess, 28), fontsize=8)

            if 0 <= roi_id < len(sd.stat):
                poly = _roi_polygon_from_stat(sd.stat[roi_id])
                if poly is not None and poly.size:
                    ax.plot(poly[:, 0], poly[:, 1], color=cmap(j), linewidth=1.6)

        fig.suptitle(f"CellRegID {cid} — ROI outlines across sessions", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)


def launch(mouse_folder: Optional[str] = None, mouse_table: Optional[str] = None):
    mf = Path(mouse_folder) if mouse_folder else None
    mt = Path(mouse_table) if mouse_table else None
    t = None
    if mf is not None or mt is not None:
        t = _load_mouse_table(mouse_folder=mf, mouse_table_path=mt)
    app = CellRegViewer(t)
    app.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-folder", type=str, default=None)
    ap.add_argument("--mouse-table", type=str, default=None)

    # parse_known_args so Interactive/Notebook injected args don't crash us
    args, _ = ap.parse_known_args()
    launch(mouse_folder=args.mouse_folder, mouse_table=args.mouse_table)


if __name__ == "__main__":
    main()

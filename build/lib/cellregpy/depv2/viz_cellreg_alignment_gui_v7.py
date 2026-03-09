\
"""
viz_cellreg_alignment_gui_v7.py

Fixes for v6 issues you hit:
1) Recursion / repeated callbacks when selecting a table row
   - Programmatic selection_set now suppresses the <<TreeviewSelect>> handler.
2) ZeroDivisionError in footprint popup when n==0
   - Guard: if no siblings, no popup.
3) fovID scalar conversion deprecation warning
   - Robust scalar extraction before int() conversion.
4) Table behavior requested:
   - Selecting a cell switches the table view to show the *siblings* (same cellRegID [+ same fovID])
     across ALL sessions, sorted by UnixTime (None/NaN safe).
   - A "Table view" dropdown lets you switch back to "Current session".

Still guaranteed:
- cellRegID is used ONLY to group siblings.
- suite2pID is used to index into the correct session's Fall.mat (stat/F/etc.) via SessionPath.

Run (Interactive):
    from viz_cellreg_alignment_gui_v7 import launch
    launch()
"""

from __future__ import annotations

import argparse
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


# ----------------------------- utilities -----------------------------

def _truncate(s: str, n: int = 34) -> str:
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

        # heuristic 1-based fix
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


# ----------------------------- data model -----------------------------

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
            bg=bg,
            fs=fs,
            F=F,
            Fneu=Fneu,
            C=C,
            S=S,
            spks=spks,
        )


# ----------------------------- GUI -----------------------------

class CellRegViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CellRegPy Viewer (validation)")
        self.geometry("1500x880")

        self.cache = SessionCache()
        self.mouse_folder: Optional[Path] = None

        self.t: Optional[pd.DataFrame] = None
        self.sessions: List[str] = []
        self.mouse_name: str = "mouse"
        self.current_session: Optional[str] = None

        self._suppress_table_event = False
        self._selected_row: Optional[int] = None
        self._selected_siblings: np.ndarray = np.array([], dtype=int)

        self.show_cells_only = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.label_mode = tk.StringVar(value="both")
        self.table_view = tk.StringVar(value="Current session")
        self.trace_kind = tk.StringVar(value="F (Suite2p)")
        self.show_fneu = tk.BooleanVar(value=False)
        self.show_spikes = tk.BooleanVar(value=True)

        self._build_widgets()

        self.fig = plt.Figure(figsize=(9.5, 7), dpi=100)
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
        ttk.Checkbutton(top, text="Labels", variable=self.show_labels, command=self.refresh).pack(side=tk.LEFT, padx=6)

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

        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.status = ttk.Label(bottom, text="File → Open Mouse Folder/Table to begin.")
        self.status.pack(side=tk.LEFT)

    def _show_help(self):
        messagebox.showinfo("Help", "Selecting a cell switches the table to show sibling rows (same cellRegID) across sessions.\nProgrammatic selection no longer triggers recursion.")

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

    def refresh(self):
        if self.t is None or self.current_session is None:
            self._draw_empty()
            return
        self._draw_session_image_and_rois()
        self._populate_tree()
        self.ax_tr.cla()
        self.ax_tr.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _draw_empty(self):
        self.ax_img.cla()
        self.ax_tr.cla()
        self.ax_img.set_axis_off()
        self.ax_tr.set_axis_off()
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

        for r in rows:
            sess = str(self.t.loc[r, "Session"])
            sess_path = Path(str(self.t.loc[r, "SessionPath"]))
            suite2p = int(self.t.loc[r, "suite2pID"])
            cellreg = int(self.t.loc[r, "cellRegID"])
            fov = self.t.loc[r, "fovID"] if "fovID" in self.t.columns else ""
            ut = self.t.loc[r, "UnixTime"] if "UnixTime" in self.t.columns else ""
            iscell = ""
            try:
                sd = self.cache.get(sess, sess_path)
                if 0 <= suite2p < sd.iscell.size:
                    iscell = "1" if bool(sd.iscell[suite2p]) else "0"
            except Exception:
                pass
            self.tree.insert("", "end", iid=str(int(r)), values=(int(r), sess, suite2p, cellreg, fov, ut, iscell))

        self.tbl_title.configure(text=f"{title} | rows={len(rows)}")

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
        session_path = Path(str(self.t.loc[rows[0], "SessionPath"]))
        sd = self.cache.get(self.current_session, session_path)
        bg = np.asarray(sd.bg, dtype=float)
        self.ax_img.imshow(bg, cmap="gray", aspect="auto")
        self.ax_img.set_title(self.current_session, fontsize=9)

        for r_i in rows:
            roi_id = int(self.t.loc[r_i, "suite2pID"])
            if roi_id < 0 or roi_id >= len(sd.stat):
                continue
            if self.show_cells_only.get():
                if not (0 <= roi_id < sd.iscell.size and bool(sd.iscell[roi_id])):
                    continue
            poly = _roi_polygon_from_stat(sd.stat[roi_id])
            if poly is None or poly.size == 0:
                continue
            patch = Polygon(poly, closed=True, fill=False, edgecolor=(0.2, 0.6, 1.0, 0.9), linewidth=0.8, picker=True)
            patch._cellreg_row_index = int(r_i)  # type: ignore[attr-defined]
            self.ax_img.add_patch(patch)
            if self.show_labels.get():
                if self.label_mode.get() == "suite2pID":
                    label = str(roi_id)
                elif self.label_mode.get() == "cellRegID":
                    label = str(int(self.t.loc[r_i, "cellRegID"]))
                else:
                    label = f"{roi_id}|{int(self.t.loc[r_i, 'cellRegID'])}"
                self.ax_img.text(float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1])), label, color="yellow", fontsize=7)

        self.ax_img.set_xlim([0, bg.shape[1]])
        self.ax_img.set_ylim([bg.shape[0], 0])

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

        self._selected_row = int(row_idx)
        cid = int(self.t.loc[row_idx, "cellRegID"])
        suite2p_sel = int(self.t.loc[row_idx, "suite2pID"])
        fov_val = _fov_scalar(self.t.loc[row_idx, "fovID"]) if "fovID" in self.t.columns else None

        if fov_val is None:
            sib_mask = (self.t["cellRegID"].astype(int) == cid)
        else:
            sib_mask = (self.t["cellRegID"].astype(int) == cid) & (pd.to_numeric(self.t["fovID"], errors="coerce") == fov_val)

        siblings = self.t.index[sib_mask].to_numpy()
        siblings = _safe_sort_rows_by_unix_time(self.t, siblings)
        self._selected_siblings = siblings

        self.table_view.set("Selected cell (siblings)")
        self.table_view_cb.set("Selected cell (siblings)")
        self._populate_tree()

        self._suppress_table_event = True
        try:
            self.tree.selection_set(str(int(row_idx)))
            self.tree.see(str(int(row_idx)))
        finally:
            self._suppress_table_event = False

        self.debug_txt.delete("1.0", tk.END)
        self.debug_txt.insert(tk.END, f"Selected row={row_idx}\ncellRegID={cid} | suite2pID={suite2p_sel} | fovID={fov_val}\n\n")
        for r in siblings:
            self.debug_txt.insert(tk.END, f"row={int(r)} | session={self.t.loc[r,'Session']} | suite2pID={int(self.t.loc[r,'suite2pID'])}\n")

        self.status.configure(text=f"Selected cellRegID={cid} (row={row_idx}) | suite2pID={suite2p_sel} | fovID={fov_val}")

        if open_popups and siblings.size > 0:
            self._popup_native_footprints(siblings, cid)

    def _popup_native_footprints(self, rows: np.ndarray, cid: int):
        assert self.t is not None
        n = int(rows.size)
        if n <= 0:
            return
        cols = max(1, int(np.ceil(np.sqrt(n))))
        nrows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(min(16, 4 * cols), min(12, 3.8 * nrows)))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for j, r in enumerate(rows):
            ax = axes[j]
            sess = str(self.t.loc[r, "Session"])
            sess_path = Path(str(self.t.loc[r, "SessionPath"]))
            roi_id = int(self.t.loc[r, "suite2pID"])
            sd = self.cache.get(sess, sess_path)
            bg = np.asarray(sd.bg, dtype=float)
            ax.imshow(bg, cmap="gray")
            ax.set_title(f"{_truncate(sess, 26)} | suite2pID={roi_id}", fontsize=8)
            if 0 <= roi_id < len(sd.stat):
                fp = _roi_footprint_image_from_stat(sd.stat[roi_id], shape_hw=bg.shape[:2])
                if fp is not None:
                    m = float(np.nanmax(fp)) if np.isfinite(np.nanmax(fp)) else 0.0
                    if m > 0:
                        ax.imshow(fp / m, cmap="viridis", alpha=0.70)
                poly = _roi_polygon_from_stat(sd.stat[roi_id])
                if poly is not None and poly.size:
                    ax.plot(poly[:, 0], poly[:, 1], color="cyan", linewidth=1.4)
        fig.suptitle(f"Spatial footprints (native) | CellRegID {cid}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)


def launch():
    app = CellRegViewer()
    app.mainloop()


if __name__ == "__main__":
    launch()

"""
cellregpy/gui_run_pipeline.py

GUI runner for CellRegPy.
- Queue one or more mouse folders
- Edit core CellRegConfig parameters (and autosimple/autoflex knobs)
- Run the pipeline sequentially (in a background thread) to avoid UI freezes
- Optional: launch the QC alignment viewer (gui_check_alignment) on results

Notes (Windows / notebooks):
- Multiprocessing can cause "recursive spawning" if called from a notebook or
  without an if __name__ == "__main__" guard. This GUI disables parallel
  processing by default (use_parallel_processing=False). You can re-enable it
  if you run this as a standalone script.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Import your package
import cellregpy


# ------------------------------ helpers ------------------------------

def _now() -> str:
    return time.strftime("%H:%M:%S")


def _as_bool(v: Any) -> bool:
    return bool(v)


def _safe_float(s: str, default: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return float(default)


def _safe_int(s: str, default: int) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return int(default)


def _open_in_file_explorer(path: Path) -> None:
    path = Path(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def _safe_setattr(obj: Any, name: str, value: Any) -> bool:
    """
    Set attribute only if it exists on obj.
    Returns True if set, False otherwise.
    """
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False


# ------------------------------ GUI ------------------------------

class CellRegBatchRunner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CellRegPy — Batch Runner")
        self.geometry("1200x780")

        # state
        self._log_q: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._last_mouse_folder: Optional[Path] = None

        # UI vars
        self.var_use_run_pipeline = tk.BooleanVar(value=True)
        self.var_export_csv = tk.BooleanVar(value=True)
        self.var_save_figures = tk.BooleanVar(value=True)
        self.var_also_pdf = tk.BooleanVar(value=True)
        self.var_figures_visibility = tk.StringVar(value="off")

        # Important config defaults
        self.var_microns_per_pixel = tk.StringVar(value="2.0")
        self.var_maximal_distance = tk.StringVar(value="14.0")
        self.var_p_same_threshold = tk.StringVar(value="0.5")
        self.var_model_type = tk.StringVar(value="auto")  # auto | Spatial correlation | Centroid distance
        self.var_dual_model = tk.BooleanVar(value=True)
        self.var_spatial_corr_floor = tk.StringVar(value="0.5")

        self.var_correlation_threshold = tk.StringVar(value="0.65")
        self.var_alignable_threshold = tk.StringVar(value="0.3")
        self.var_maximal_rotation = tk.StringVar(value="30.0")
        self.var_alignment_type = tk.StringVar(value="translations_and_rotations")
        # IMPORTANT: disable parallel by default to avoid recursion/spawn loops in interactive contexts
        self.var_use_parallel_processing = tk.BooleanVar(value=False)

        # Autosimple (median corr)
        self.var_auto_simple_on = tk.BooleanVar(value=False)
        self.var_auto_simple_raw = tk.StringVar(value="0.90")
        self.var_auto_simple_aligned = tk.StringVar(value="0.95")
        self.var_auto_simple_method = tk.StringVar(value="centroid_hungarian")

        # Autoflex (peak corr)
        self.var_auto_flex_on = tk.BooleanVar(value=True)
        self.var_auto_flex_peak = tk.StringVar(value="0.95")

        # Simple assignment params
        self.var_simple_mask_thr = tk.StringVar(value="0.15")
        self.var_simple_iou_thr = tk.StringVar(value="0.10")
        self.var_simple_dist_thr_um = tk.StringVar(value="6.0")
        self.var_simple_cost_beta = tk.StringVar(value="0.25")

        # Test run
        self.var_test_run = tk.BooleanVar(value=False)
        self.var_test_run_type = tk.StringVar(value="test random alignment")

        self._build_widgets()
        self.after(100, self._drain_log_q)

    # ---------------- UI layout ----------------

    def _build_widgets(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # LEFT: mouse folder queue
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Mouse folders (queue)", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.lst = tk.Listbox(left, height=16, selectmode=tk.EXTENDED)
        self.lst.pack(fill=tk.BOTH, expand=False, pady=(6, 6))

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X)

        ttk.Button(btns, text="Add folder…", command=self._add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Add many…", command=self._add_many_dialog).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Remove selected", command=self._remove_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btns, text="Clear", command=self._clear_queue).pack(side=tk.LEFT, padx=(0, 6))

        # RUN buttons
        runbar = ttk.Frame(left)
        runbar.pack(fill=tk.X, pady=(10, 0))
        self.btn_run = ttk.Button(runbar, text="▶ Run queue", command=self._run_queue)
        self.btn_run.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_stop = ttk.Button(runbar, text="■ Stop (after current)", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(runbar, text="Open last results folder", command=self._open_last_results).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(runbar, text="Open QC Viewer for last", command=self._open_viewer_last).pack(side=tk.LEFT)

        # RIGHT: params + log
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        nb = ttk.Notebook(right)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_core = ttk.Frame(nb)
        tab_align = ttk.Frame(nb)
        tab_auto = ttk.Frame(nb)
        tab_io = ttk.Frame(nb)
        tab_log = ttk.Frame(nb)

        nb.add(tab_core, text="Core")
        nb.add(tab_align, text="Alignment")
        nb.add(tab_auto, text="Auto-switch")
        nb.add(tab_io, text="I/O")
        nb.add(tab_log, text="Log")

        # --- Core tab ---
        self._grid_entry(tab_core, 0, "microns_per_pixel", self.var_microns_per_pixel)
        self._grid_entry(tab_core, 1, "maximal_distance (µm)", self.var_maximal_distance)
        self._grid_entry(tab_core, 2, "p_same_threshold", self.var_p_same_threshold)
        self._grid_combo(tab_core, 3, "model_type", self.var_model_type, ["auto", "Spatial correlation", "Centroid distance"])
        self._grid_check(tab_core, 4, "dual_model (centroid + spatial floor)", self.var_dual_model)
        self._grid_entry(tab_core, 5, "spatial_corr_floor", self.var_spatial_corr_floor)

        self._grid_check(tab_core, 7, "test_run (limit sessions)", self.var_test_run)
        self._grid_combo(tab_core, 8, "test_run_type", self.var_test_run_type,
                         ["test random alignment", "test difficult alignment", "first 4 sessions"])

        # --- Alignment tab ---
        self._grid_entry(tab_align, 0, "correlation_threshold (alignability)", self.var_correlation_threshold)
        self._grid_entry(tab_align, 1, "alignable_threshold (graph)", self.var_alignable_threshold)
        self._grid_combo(tab_align, 2, "alignment_type", self.var_alignment_type,
                         ["translations", "translations_and_rotations", "non_rigid"])
        self._grid_entry(tab_align, 3, "maximal_rotation (deg)", self.var_maximal_rotation)
        self._grid_check(tab_align, 5, "use_parallel_processing (⚠ can recurse in notebooks)", self.var_use_parallel_processing)

        # --- Auto tab ---
        self._grid_check(tab_auto, 0, "auto_simple_on_high_similarity (median corr)", self.var_auto_simple_on)
        self._grid_entry(tab_auto, 1, "auto_simple_raw_corr_threshold", self.var_auto_simple_raw)
        self._grid_entry(tab_auto, 2, "auto_simple_aligned_corr_threshold", self.var_auto_simple_aligned)
        self._grid_combo(tab_auto, 3, "auto_simple_method",
                         self.var_auto_simple_method, ["iou_hungarian", "centroid_hungarian"])

        ttk.Separator(tab_auto).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        self._grid_check(tab_auto, 6, "auto_flex_on_high_peak (peak corr)", self.var_auto_flex_on)
        self._grid_entry(tab_auto, 7, "auto_flex_peak_threshold", self.var_auto_flex_peak)

        ttk.Label(tab_auto, text="Simple-mode params (used by IoU/Hungarian & centroid/Hungarian)",
                  font=("Segoe UI", 9, "bold")).grid(row=9, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._grid_entry(tab_auto, 10, "simple_mask_threshold (IoU)", self.var_simple_mask_thr)
        self._grid_entry(tab_auto, 11, "simple_iou_threshold (IoU)", self.var_simple_iou_thr)
        self._grid_entry(tab_auto, 12, "simple_dist_threshold_um (centroid)", self.var_simple_dist_thr_um)
        self._grid_entry(tab_auto, 13, "simple_cost_beta", self.var_simple_cost_beta)

        # --- I/O tab ---
        self._grid_check(tab_io, 0, "Use run_pipeline() (dual-model enforced inside)", self.var_use_run_pipeline)
        self._grid_check(tab_io, 1, "export_csv (mouse_table.csv)", self.var_export_csv)
        self._grid_check(tab_io, 2, "save_figures", self.var_save_figures)
        self._grid_check(tab_io, 3, "also_pdf", self.var_also_pdf)
        self._grid_combo(tab_io, 4, "figures_visibility", self.var_figures_visibility, ["off", "on"])

        # preset buttons
        pb = ttk.Frame(tab_io)
        pb.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(pb, text="Save preset…", command=self._save_preset).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(pb, text="Load preset…", command=self._load_preset).pack(side=tk.LEFT, padx=(0, 6))

        # --- Log tab ---
        self.txt = tk.Text(tab_log, wrap="none", height=20)
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.insert("end", f"[{_now()}] ready\n")
        self.txt.configure(state="disabled")

    def _grid_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(6, 12), pady=6)
        ent = ttk.Entry(parent, textvariable=var, width=28)
        ent.grid(row=row, column=1, sticky="w", padx=6, pady=6)

    def _grid_check(self, parent: ttk.Frame, row: int, label: str, var: tk.BooleanVar):
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=6)

    def _grid_combo(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, values: List[str]):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(6, 12), pady=6)
        cb = ttk.Combobox(parent, textvariable=var, values=values, width=26, state="readonly")
        cb.grid(row=row, column=1, sticky="w", padx=6, pady=6)

    # ---------------- queue ops ----------------

    def _add_folder(self):
        d = filedialog.askdirectory(title="Select mouse folder (contains session folders)")
        if not d:
            return
        self.lst.insert("end", d)

    def _add_many_dialog(self):
        """
        Allow user to paste many folders (newline-separated).
        """
        win = tk.Toplevel(self)
        win.title("Add many mouse folders")
        win.geometry("780x420")
        ttk.Label(win, text="Paste one folder per line:").pack(anchor="w", padx=8, pady=6)
        txt = tk.Text(win, wrap="none")
        txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        def _add():
            lines = [ln.strip() for ln in txt.get("1.0", "end").splitlines()]
            lines = [ln for ln in lines if ln]
            added = 0
            for ln in lines:
                self.lst.insert("end", ln)
                added += 1
            self._log(f"added {added} folder(s)")
            win.destroy()

        ttk.Button(win, text="Add", command=_add).pack(side=tk.RIGHT, padx=8, pady=8)

    def _remove_selected(self):
        sel = list(self.lst.curselection())
        if not sel:
            return
        for i in reversed(sel):
            self.lst.delete(i)

    def _clear_queue(self):
        self.lst.delete(0, "end")

    def _get_queue(self) -> List[Path]:
        items = [self.lst.get(i) for i in range(self.lst.size())]
        out = []
        for it in items:
            p = Path(str(it))
            if p.exists():
                out.append(p)
            else:
                self._log(f"WARNING: path does not exist: {p}")
        return out

    # ---------------- presets ----------------

    def _get_ui_state(self) -> Dict[str, Any]:
        # anything JSON-serializable
        return {
            "use_run_pipeline": bool(self.var_use_run_pipeline.get()),
            "export_csv": bool(self.var_export_csv.get()),
            "save_figures": bool(self.var_save_figures.get()),
            "also_pdf": bool(self.var_also_pdf.get()),
            "figures_visibility": str(self.var_figures_visibility.get()),

            "microns_per_pixel": str(self.var_microns_per_pixel.get()),
            "maximal_distance": str(self.var_maximal_distance.get()),
            "p_same_threshold": str(self.var_p_same_threshold.get()),
            "model_type": str(self.var_model_type.get()),
            "dual_model": bool(self.var_dual_model.get()),
            "spatial_corr_floor": str(self.var_spatial_corr_floor.get()),

            "correlation_threshold": str(self.var_correlation_threshold.get()),
            "alignable_threshold": str(self.var_alignable_threshold.get()),
            "maximal_rotation": str(self.var_maximal_rotation.get()),
            "alignment_type": str(self.var_alignment_type.get()),
            "use_parallel_processing": bool(self.var_use_parallel_processing.get()),

            "auto_simple_on": bool(self.var_auto_simple_on.get()),
            "auto_simple_raw": str(self.var_auto_simple_raw.get()),
            "auto_simple_aligned": str(self.var_auto_simple_aligned.get()),
            "auto_simple_method": str(self.var_auto_simple_method.get()),

            "auto_flex_on": bool(self.var_auto_flex_on.get()),
            "auto_flex_peak": str(self.var_auto_flex_peak.get()),

            "simple_mask_threshold": str(self.var_simple_mask_thr.get()),
            "simple_iou_threshold": str(self.var_simple_iou_thr.get()),
            "simple_dist_threshold_um": str(self.var_simple_dist_thr_um.get()),
            "simple_cost_beta": str(self.var_simple_cost_beta.get()),

            "test_run": bool(self.var_test_run.get()),
            "test_run_type": str(self.var_test_run_type.get()),
        }

    def _apply_ui_state(self, st: Dict[str, Any]):
        def sget(k, default=None): return st.get(k, default)

        self.var_use_run_pipeline.set(bool(sget("use_run_pipeline", True)))
        self.var_export_csv.set(bool(sget("export_csv", True)))
        self.var_save_figures.set(bool(sget("save_figures", True)))
        self.var_also_pdf.set(bool(sget("also_pdf", True)))
        self.var_figures_visibility.set(str(sget("figures_visibility", "off")))

        self.var_microns_per_pixel.set(str(sget("microns_per_pixel", "2.0")))
        self.var_maximal_distance.set(str(sget("maximal_distance", "14.0")))
        self.var_p_same_threshold.set(str(sget("p_same_threshold", "0.5")))
        self.var_model_type.set(str(sget("model_type", "auto")))
        self.var_dual_model.set(bool(sget("dual_model", True)))
        self.var_spatial_corr_floor.set(str(sget("spatial_corr_floor", "0.5")))

        self.var_correlation_threshold.set(str(sget("correlation_threshold", "0.65")))
        self.var_alignable_threshold.set(str(sget("alignable_threshold", "0.3")))
        self.var_maximal_rotation.set(str(sget("maximal_rotation", "30.0")))
        self.var_alignment_type.set(str(sget("alignment_type", "translations_and_rotations")))
        self.var_use_parallel_processing.set(bool(sget("use_parallel_processing", False)))

        self.var_auto_simple_on.set(bool(sget("auto_simple_on", False)))
        self.var_auto_simple_raw.set(str(sget("auto_simple_raw", "0.90")))
        self.var_auto_simple_aligned.set(str(sget("auto_simple_aligned", "0.95")))
        self.var_auto_simple_method.set(str(sget("auto_simple_method", "centroid_hungarian")))

        self.var_auto_flex_on.set(bool(sget("auto_flex_on", True)))
        self.var_auto_flex_peak.set(str(sget("auto_flex_peak", "0.95")))

        self.var_simple_mask_thr.set(str(sget("simple_mask_threshold", "0.15")))
        self.var_simple_iou_thr.set(str(sget("simple_iou_threshold", "0.10")))
        self.var_simple_dist_thr_um.set(str(sget("simple_dist_threshold_um", "6.0")))
        self.var_simple_cost_beta.set(str(sget("simple_cost_beta", "0.25")))

        self.var_test_run.set(bool(sget("test_run", False)))
        self.var_test_run_type.set(str(sget("test_run_type", "test random alignment")))

    def _save_preset(self):
        st = self._get_ui_state()
        f = filedialog.asksaveasfilename(
            title="Save preset (json)",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not f:
            return
        try:
            Path(f).write_text(json.dumps(st, indent=2))
            self._log(f"saved preset: {f}")
        except Exception as e:
            messagebox.showerror("Save preset failed", str(e))

    def _load_preset(self):
        f = filedialog.askopenfilename(
            title="Load preset (json)",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not f:
            return
        try:
            st = json.loads(Path(f).read_text())
            self._apply_ui_state(st)
            self._log(f"loaded preset: {f}")
        except Exception as e:
            messagebox.showerror("Load preset failed", str(e))

    # ---------------- logging ----------------

    def _log(self, msg: str):
        self._log_q.put(f"[{_now()}] {msg}")

    def _drain_log_q(self):
        try:
            drained = 0
            while True:
                try:
                    line = self._log_q.get_nowait()
                except Exception:
                    break
                self.txt.configure(state="normal")
                self.txt.insert("end", line + "\n")
                self.txt.see("end")
                self.txt.configure(state="disabled")
                drained += 1
                if drained > 200:
                    break
        finally:
            self.after(100, self._drain_log_q)

    # ---------------- cfg build ----------------

    def _build_cfg(self) -> cellregpy.CellRegConfig:
        cfg = cellregpy.CellRegConfig()

        # core
        _safe_setattr(cfg, "microns_per_pixel", _safe_float(self.var_microns_per_pixel.get(), cfg.microns_per_pixel))
        _safe_setattr(cfg, "maximal_distance", _safe_float(self.var_maximal_distance.get(), cfg.maximal_distance))
        _safe_setattr(cfg, "p_same_threshold", _safe_float(self.var_p_same_threshold.get(), cfg.p_same_threshold))
        _safe_setattr(cfg, "model_type", str(self.var_model_type.get()).strip())
        _safe_setattr(cfg, "dual_model", bool(self.var_dual_model.get()))
        _safe_setattr(cfg, "apply_spatial_floor_filter", bool(self.var_dual_model.get()))
        _safe_setattr(cfg, "spatial_corr_floor", _safe_float(self.var_spatial_corr_floor.get(), getattr(cfg, "spatial_corr_floor", 0.5)))

        # alignment
        _safe_setattr(cfg, "correlation_threshold", _safe_float(self.var_correlation_threshold.get(), cfg.correlation_threshold))
        _safe_setattr(cfg, "alignable_threshold", _safe_float(self.var_alignable_threshold.get(), getattr(cfg, "alignable_threshold", 0.3)))
        _safe_setattr(cfg, "alignment_type", str(self.var_alignment_type.get()).strip())
        _safe_setattr(cfg, "maximal_rotation", _safe_float(self.var_maximal_rotation.get(), getattr(cfg, "maximal_rotation", 30.0)))
        _safe_setattr(cfg, "use_parallel_processing", bool(self.var_use_parallel_processing.get()))

        # figures
        _safe_setattr(cfg, "save_figures", bool(self.var_save_figures.get()))
        _safe_setattr(cfg, "also_pdf", bool(self.var_also_pdf.get()))
        _safe_setattr(cfg, "figures_visibility", str(self.var_figures_visibility.get()).strip())
        _safe_setattr(cfg, "close_figures", True)

        # test run
        _safe_setattr(cfg, "test_run", bool(self.var_test_run.get()))
        _safe_setattr(cfg, "test_run_type", str(self.var_test_run_type.get()).strip())

        # autosimple
        _safe_setattr(cfg, "auto_simple_on_high_similarity", bool(self.var_auto_simple_on.get()))
        _safe_setattr(cfg, "auto_simple_raw_corr_threshold", _safe_float(self.var_auto_simple_raw.get(), getattr(cfg, "auto_simple_raw_corr_threshold", 0.90)))
        _safe_setattr(cfg, "auto_simple_aligned_corr_threshold", _safe_float(self.var_auto_simple_aligned.get(), getattr(cfg, "auto_simple_aligned_corr_threshold", 0.95)))
        _safe_setattr(cfg, "auto_simple_method", str(self.var_auto_simple_method.get()).strip())

        # autoflex
        _safe_setattr(cfg, "auto_flex_on_high_peak", bool(self.var_auto_flex_on.get()))
        _safe_setattr(cfg, "auto_flex_peak_threshold", _safe_float(self.var_auto_flex_peak.get(), getattr(cfg, "auto_flex_peak_threshold", 0.95)))

        # simple params
        _safe_setattr(cfg, "simple_mask_threshold", _safe_float(self.var_simple_mask_thr.get(), getattr(cfg, "simple_mask_threshold", 0.15)))
        _safe_setattr(cfg, "simple_iou_threshold", _safe_float(self.var_simple_iou_thr.get(), getattr(cfg, "simple_iou_threshold", 0.10)))
        _safe_setattr(cfg, "simple_dist_threshold_um", _safe_float(self.var_simple_dist_thr_um.get(), getattr(cfg, "simple_dist_threshold_um", 6.0)))
        _safe_setattr(cfg, "simple_cost_beta", _safe_float(self.var_simple_cost_beta.get(), getattr(cfg, "simple_cost_beta", 0.25)))

        # guard: if running in an interactive context, disable parallel by default
        if not self.var_use_parallel_processing.get():
            _safe_setattr(cfg, "use_parallel_processing", False)

        return cfg

    # ---------------- run actions ----------------

    def _set_running(self, running: bool):
        self.btn_run.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _stop(self):
        self._stop_flag.set()
        self._log("stop requested (will stop after current mouse finishes)")

    def _run_queue(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Already running", "A run is already in progress.")
            return

        folders = self._get_queue()
        if not folders:
            messagebox.showwarning("No folders", "Add at least one mouse folder to the queue.")
            return

        cfg = self._build_cfg()

        # sanity: warn about parallel
        if bool(getattr(cfg, "use_parallel_processing", False)):
            if not messagebox.askyesno(
                "Parallel processing enabled",
                "use_parallel_processing=True can cause recursive spawning in notebooks/interactive.\n\n"
                "If you run this GUI as a standalone script, it's usually fine.\n\nContinue?"
            ):
                return

        self._stop_flag.clear()
        self._set_running(True)
        self._log(f"starting run for {len(folders)} mouse folder(s)")

        self._worker = threading.Thread(target=self._worker_run, args=(folders, cfg), daemon=True)
        self._worker.start()

    def _worker_run(self, folders: List[Path], cfg: cellregpy.CellRegConfig):
        use_run_pipeline = bool(self.var_use_run_pipeline.get())
        export_csv = bool(self.var_export_csv.get())
        save_figures = bool(self.var_save_figures.get())
        figures_visibility = str(self.var_figures_visibility.get()).strip()
        spatial_corr_floor = _safe_float(self.var_spatial_corr_floor.get(), getattr(cfg, "spatial_corr_floor", 0.5))

        for i, folder in enumerate(folders, start=1):
            if self._stop_flag.is_set():
                break

            folder = Path(folder)
            self._log(f"[{i}/{len(folders)}] running: {folder}")
            t0 = time.perf_counter()

            try:
                cfg_i = copy.deepcopy(cfg)

                if use_run_pipeline:
                    mouse_table, mouse_data = cellregpy.run_pipeline(
                        folder,
                        cfg=cfg_i,
                        spatial_corr_floor=spatial_corr_floor,
                        save_figures=save_figures,
                        figures_visibility=figures_visibility,
                        export_csv=export_csv,
                    )
                else:
                    cr = cellregpy.CellRegPy(cfg_i)
                    mouse_table, mouse_data = cr.run([folder])

                dt = time.perf_counter() - t0
                self._last_mouse_folder = folder
                # basic summary
                nrows = getattr(mouse_table, "shape", (0, 0))[0] if mouse_table is not None else 0
                nsess = int(mouse_table["Session"].nunique()) if (mouse_table is not None and "Session" in mouse_table.columns) else 0
                ncellreg = int(mouse_table["cellRegID"].nunique()) if (mouse_table is not None and "cellRegID" in mouse_table.columns) else 0
                self._log(f"done in {dt:.1f}s | sessions={nsess} | unique cellRegID={ncellreg} | rows={nrows}")
            except Exception as e:
                dt = time.perf_counter() - t0
                self._log(f"ERROR after {dt:.1f}s: {e}")
                self._log(traceback.format_exc())

        self._log("run finished")
        self.after(0, lambda: self._set_running(False))

    # ---------------- convenience buttons ----------------

    def _open_last_results(self):
        if self._last_mouse_folder is None:
            messagebox.showinfo("No runs yet", "Run at least one mouse first.")
            return
        results_dir = self._last_mouse_folder / "1_CellReg"
        if results_dir.exists():
            _open_in_file_explorer(results_dir)
        else:
            _open_in_file_explorer(self._last_mouse_folder)

    def _open_viewer_last(self):
        if self._last_mouse_folder is None:
            messagebox.showinfo("No runs yet", "Run at least one mouse first.")
            return
        self._open_viewer_for_folder(self._last_mouse_folder)

    def _open_viewer_for_folder(self, mouse_folder: Path):
        """
        Launch the QC viewer (gui_check_alignment) in a separate Python process,
        pre-loaded to the given mouse folder.
        """
        mouse_folder = Path(mouse_folder)

        # Build a tiny inline launcher so we can call CellRegViewer.set_table(...) without modifying the viewer.
        code = f"""
from pathlib import Path
from cellregpy.gui_check_alignment import CellRegViewer, _load_mouse_table
t = _load_mouse_table(mouse_folder=Path(r'{str(mouse_folder)}'))
app = CellRegViewer()
app.set_table(t)
app.mainloop()
"""
        try:
            subprocess.Popen([sys.executable, "-c", code], cwd=str(mouse_folder))
            self._log(f"opened QC viewer for: {mouse_folder}")
        except Exception as e:
            messagebox.showerror("Open viewer failed", str(e))


def launch():
    app = CellRegBatchRunner()
    app.mainloop()


if __name__ == "__main__":
    launch()

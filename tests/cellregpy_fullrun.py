from pathlib import Path
import sys

# If you are running this notebook from the repo root, keep this.
# Otherwise, set this to the folder that contains cellregpy.py
repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))

from cellregpy import CellRegConfig, run_pipeline

# define datafolder to run over
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    sessions_to_cellreg = [
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L6REstim\L629_M_LeftPFC_L6REChrimson_Panrec"),
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L6REstim\L631_M_LeftPFC_L6REChrimson_Panrec"),
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L5\AB13_M_LeftPFC_L6Chrimson_L5CTrec-ConFoffGCaMP6f"),
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L5\L605_M_RightPFC_L6Chrimson_PFC-L5CTrec"),
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L5\L623_M_ConFoffGCaMP_L6Chrimson_L5CTrec"),
        #Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_L5\L632_F_LeftPFC_L6Chrimson_L5CTrec-FLEXgcamp6f"),
        Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_CC\A10_F_LeftPFC_L6Chrimson_CCrec"),
       # Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_CC\L607T4_M_RightPFC_L6Chrimson_PFC-CCrec"),
       # Path(r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_CC\TA05_M_RightPFC_L6Chrimson_CCrec-ConFoffGCaMP6f"),
    ]

    # get all subdirs
    cfg = CellRegConfig()
    cfg.use_parallel_processing = False
    for mouse_folder in sessions_to_cellreg:
        mouse_table, mouse_data = run_pipeline(
            mouse_folder,
            cfg=cfg,
            spatial_corr_floor=cfg.spatial_corr_floor,
            save_figures=cfg.save_figures,
            figures_visibility='off',
            export_csv=True,
        )
    plt.close('all')

if __name__ == "__main__":
    main()
# save out EMD cleaned signal
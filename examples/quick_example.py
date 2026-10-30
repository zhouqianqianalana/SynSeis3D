#!/usr/bin/env python3
"""
Quick example: generate one labeled synthetic seismic volume and plot it.

Runs the full 7-step pipeline for a single model using config.yaml, writes the
seismic volume and its four co-registered labels, and saves a figure showing
the section Y = grid/2 of each. The defaults (256^3, seed 42, 6 interval
boundaries, 4 channels, 3 faults) reproduce the model shown in Figs. 1-8 of
the paper.

    python examples/quick_example.py                 # 256^3, ~10 s
    python examples/quick_example.py --grid 128      # smaller and faster
    python examples/quick_example.py --seed 7 --out outputs/example_seed7

Outputs (in --out):
    DL_Input_Seismic.npy, DL_Label_{Faults,Channels,Basement,Intervals}.npy,
    config_run.yaml, metadata_step*.json, DL_Visualization_Y*.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from batch_generate import run_pipeline, extract_dl_labels  # noqa: E402
from visualize_dl_dataset import visualize_dl_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--grid", type=int, default=256, help="cubic grid size (default 256)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/example")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["grid"].update(nx=args.grid, ny=args.grid, nz=args.grid)
    cfg["random_seed"] = args.seed
    cfg["intervals"]["num_intervals"] = 6
    cfg["geobodies"]["channels"]["num_channels"] = 4
    cfg["deformation"]["faults"]["num_faults"] = 3

    out_dir = Path(args.out)
    cfg["output"] = {"output_dir": str(out_dir.parent), "model_name": out_dir.name,
                     "preserve_debug_figures": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config_run.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    print(f"Generating a {args.grid}^3 model (seed {args.seed}) ...")
    store = {}
    out7, timings = run_pipeline(cfg_path, store=store)
    extract_dl_labels(out7, out_dir, store=store)
    print(f"Pipeline finished in {sum(timings.values()):.1f} s")

    seis = np.load(out_dir / "DL_Input_Seismic.npy")
    print(f"Seismic: shape {seis.shape}, RMS {np.sqrt(np.mean(seis ** 2)):.3f}")
    for name in ("Faults", "Channels", "Basement"):
        lab = np.load(out_dir / f"DL_Label_{name}.npy")
        print(f"{name:9s}: {100 * lab.mean():5.2f} % of voxels")
    ints = np.load(out_dir / "DL_Label_Intervals.npy")
    print(f"Intervals: classes {sorted(np.unique(ints).tolist())} (0 = basement)")

    visualize_dl_model(str(out_dir), slice_index=args.grid // 2, axis="y")


if __name__ == "__main__":
    main()

import argparse
import os
import time
import copy

import yaml
import numpy as np
from pathlib import Path

# Import the pipeline
from geomodel import (
    generate_basement_step1,
    generate_intervals_step2,
    generate_architecture_step3,
    generate_geobodies_step4,
    generate_structural_deformation_step5,
    generate_rock_properties_step6,
    generate_seismic_step7,
    get_array,
)

STEP_FUNCS = [
    ("step1_basement", generate_basement_step1),
    ("step2_intervals", generate_intervals_step2),
    ("step3_architecture", generate_architecture_step3),
    ("step4_geobodies", generate_geobodies_step4),
    ("step5_deformation", generate_structural_deformation_step5),
    ("step6_properties", generate_rock_properties_step6),
    ("step7_seismic", generate_seismic_step7),
]


def run_pipeline(config_path, store=None, verbose=True):
    """Run all 7 steps for one config. Returns (final_out_dict, per_step_seconds)."""
    timings = {}
    out = None
    for name, fn in STEP_FUNCS:
        t0 = time.time()
        if out is None:
            out = fn(config_path, store=store)
        else:
            out = fn(config_path, out, store=store)
        timings[name] = time.time() - t0
        if verbose:
            print(f"    {name}: {timings[name]:.1f}s")
    return out, timings


def extract_dl_labels(out7, model_dir, store=None):
    """Derive and save the five DL_* products from the final pipeline outputs."""
    bed_vol = get_array(store, out7["faulted_bed_id_vol"])
    fault_vol = get_array(store, out7["fault_plane_vol"])

    # Label 1: Faults (Binary: 0=Background, 1=Fault)
    dl_faults = fault_vol.astype(np.uint8)

    # Label 2: Basement (Binary: 0=Sediment, 1=Basement)
    dl_basement = np.where((bed_vol == 10) | (bed_vol == 11), 1, 0).astype(np.uint8)

    # Label 3: Channels (Binary: 0=Background, 1=Channel)
    dl_channels = np.where(bed_vol >= 50000, 1, 0).astype(np.uint8)

    # Label 4: Intervals (Categorical: 0=Basement, 1=Int1, 2=Int2, etc.)
    dl_intervals = np.where(bed_vol < 50000, bed_vol // 1000, 0)
    dl_intervals = np.where(dl_basement == 1, 0, dl_intervals).astype(np.uint8)

    np.save(model_dir / "DL_Input_Seismic.npy", get_array(store, out7["seismic_vol"]).astype(np.float32))
    np.save(model_dir / "DL_Label_Faults.npy", dl_faults)
    np.save(model_dir / "DL_Label_Basement.npy", dl_basement)
    np.save(model_dir / "DL_Label_Channels.npy", dl_channels)
    np.save(model_dir / "DL_Label_Intervals.npy", dl_intervals)


def draw_model_config(master_rng, base_cfg):
    """
    Draw one model's randomized parameters from the master RNG.

    The draw order is FIXED (intervals, channels, faults, seed [, seismic
    block]) so that fast-forwarding by model index reproduces any model
    regardless of which process generates it. The seismic block is drawn only
    when the base config sets `batch_randomize_seismic: true`, so legacy
    configs reproduce the historical stream exactly.
    """
    cfg = copy.deepcopy(base_cfg)

    cfg["intervals"] = cfg.get("intervals", {})
    cfg["intervals"]["num_intervals"] = int(master_rng.integers(3, 8))

    cfg["geobodies"] = cfg.get("geobodies", {"channels": {}})
    cfg["geobodies"]["channels"]["num_channels"] = int(master_rng.integers(1, 7))

    cfg["deformation"] = cfg.get("deformation", {"faults": {}})
    cfg["deformation"]["faults"]["num_faults"] = int(master_rng.integers(0, 5))

    cfg["random_seed"] = int(master_rng.integers(1, 999999))

    if bool(base_cfg.get("batch_randomize_seismic", False)):
        seis = cfg.setdefault("seismic", {})
        seis["noise_level"] = float(master_rng.uniform(0.08, 0.30))
        seis["lateral_footprint_voxels"] = float(master_rng.uniform(
            *base_cfg.get("batch_footprint_range", [0.5, 2.0])))
        seis["amplitude_taper_factor"] = float(master_rng.uniform(1.0, 2.2))
        scale = float(master_rng.uniform(0.7, 1.3))
        base_widths = seis.get("wavelet_widths_voxels", [3.0, 8.0, 20.0])
        seis["wavelet_widths_voxels"] = [float(w) * scale for w in base_widths]

    return cfg


def run_batch_generation(base_config_path="config.yaml", num_models=230,
                         output_root="outputs/dataset",
                         start_index=0, keep_intermediates=False):
    """
    Generate `num_models` randomized models under `output_root`.

    By default the pipeline runs fully in memory (store dict) and only the
    five DL_* products, the per-model config, and the per-step metadata JSONs
    are written to disk. Pass keep_intermediates=True to restore the legacy
    behavior where every intermediate volume is written to .npy (and then
    cleaned up after label extraction).

    `start_index` lets a SLURM array job partition the model range:
    the master RNG is advanced deterministically so model_0057 has the same
    parameters no matter which array task generates it.
    """
    out_root_path = Path(output_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    # Load the base config to act as our template
    with open(base_config_path, "r", encoding="utf-8") as fh:
        base_cfg = yaml.safe_load(fh)

    print(f"Starting Batch Generation of {num_models} models (index {start_index}..{start_index + num_models - 1})...")
    start_time = time.time()

    # Independent Master Generator for the loop. Fast-forward uses the same
    # draw function as generation, so any model index is reproducible
    # independently of which process generates it.
    master_rng = np.random.default_rng(42)
    for _ in range(start_index):
        draw_model_config(master_rng, base_cfg)

    for i in range(start_index, start_index + num_models):
        model_name = f"model_{i:04d}"
        model_dir = out_root_path / model_name

        print(f"\n--- Generating {model_name} ---")

        # 1. Randomize the parameters for this specific model
        cfg = draw_model_config(master_rng, base_cfg)

        # Ensure output points to the new specific folder
        if "output" not in cfg:
            cfg["output"] = {}
        cfg["output"]["output_dir"] = str(out_root_path)
        cfg["output"]["model_name"] = model_name

        # Forbid Step 1 from deleting the folder and destroying our config
        cfg["output"]["preserve_debug_figures"] = True

        # Save the specific config for this run
        model_dir.mkdir(parents=True, exist_ok=True)
        temp_config_path = model_dir / "config_run.yaml"
        with open(temp_config_path, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh)

        # 2. Run the Pipeline
        store = None if keep_intermediates else {}
        try:
            model_t0 = time.time()
            out7, _ = run_pipeline(temp_config_path, store=store)

            # 3. Extract Clean Deep Learning Labels
            print("Extracting Deep Learning Masks...")
            extract_dl_labels(out7, model_dir, store=store)

            # 4. Cleanup (legacy mode only: in-memory mode wrote no intermediates)
            if keep_intermediates:
                print("Cleaning up intermediate files to save memory...")
                for file in model_dir.glob("*.npy"):
                    if not file.name.startswith("DL_"):
                        os.remove(file)

            print(f"Model {model_name} completed in {time.time() - model_t0:.1f}s.")

        except Exception as e:
            print(f"FAILED on {model_name}. Error: {e}")
            continue
        finally:
            if store is not None:
                store.clear()

    total_time = time.time() - start_time
    print(f"\nBatch Generation Complete!")
    print(f"Generated {num_models} models in {total_time/60:.2f} minutes.")
    print(f"Dataset saved to: {out_root_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-generate synthetic seismic models.")
    parser.add_argument("--config", default="config.yaml", help="Base YAML config (default: config.yaml)")
    parser.add_argument("--num-models", type=int, default=230, help="Number of models to generate")
    parser.add_argument("--output-dir", default="outputs/dataset", help="Dataset root directory")
    parser.add_argument("--start-index", type=int, default=0,
                        help="First model index (for SLURM array partitioning)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Legacy mode: write every intermediate volume to .npy")
    args = parser.parse_args()

    run_batch_generation(
        base_config_path=args.config,
        num_models=args.num_models,
        output_root=args.output_dir,
        start_index=args.start_index,
        keep_intermediates=args.keep_intermediates,
    )

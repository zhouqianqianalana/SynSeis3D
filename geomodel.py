from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover
    def gaussian_filter(arr: np.ndarray, sigma: float | tuple[float, ...]) -> np.ndarray:
        """Fallback no-op filter when scipy is unavailable."""
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# In-memory pipeline store
#
# Every step communicates through {name: path} dicts. Historically each array
# was written to .npy and re-loaded by the next step; at 512^3 that is tens of
# GB of disk round-trips for intermediates that batch_generate.py deletes at
# the end anyway. When a dict is passed as `store`, arrays are kept in memory
# under their path string instead, and only explicitly-exported files (the
# DL_* products written by batch_generate.py) touch the disk. Passing
# store=None preserves the original file-based behavior exactly.
#
# Arrays handed back by get_array are NOT copied: the pipeline's step order
# never re-reads a key after a later step has mutated that array in place,
# so the in-memory and file-based pipelines produce identical volumes.
# ─────────────────────────────────────────────────────────────────────────────

def put_array(store: dict | None, path: Path | str, arr: np.ndarray) -> None:
    if store is None:
        np.save(path, arr)
    else:
        store[str(path)] = arr


def get_array(store: dict | None, path: Path | str) -> np.ndarray:
    if store is not None:
        key = str(path)
        if key in store:
            return store[key]
    return np.load(path)


def _smooth_noise_2d(rng: np.random.Generator, nx: int, ny: int, sigma_xy: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=(nx, ny)).astype(np.float32)
    if sigma_xy <= 0:
        return raw
    out = gaussian_filter(raw, sigma=sigma_xy)
    std = float(np.std(out))
    if std < 1e-8:
        return np.zeros((nx, ny), dtype=np.float32)
    return (out / std).astype(np.float32)


def _pick(rng: np.random.Generator, val: Any, cast=float):
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo, hi = val
        if cast is int:
            return int(rng.integers(int(lo), int(hi) + 1))
        return cast(rng.uniform(float(lo), float(hi)))
    return cast(val)


def _build_top_basement(cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    grid = cfg["grid"]
    bcfg = cfg["basement"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dz_m = float(grid["dz_m"])

    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, nx, dtype=np.float32),
        np.linspace(0.0, 1.0, ny, dtype=np.float32),
        indexing="ij",
    )

    top_frac = _pick(rng, bcfg.get("top_depth_fraction", [0.42, 0.82]), float)
    trend_x = _pick(rng, bcfg.get("trend_x_samples", [-14.0, 14.0]), float)
    trend_y = _pick(rng, bcfg.get("trend_y_samples", [-10.0, 10.0]), float)
    trend = top_frac * nz + trend_x * (xx - 0.5) + trend_y * (yy - 0.5)

    amp_s = _pick(rng, bcfg.get("relief_amplitude_m", [120.0, 280.0]), float) / max(dz_m, 1e-6)
    sigma_long = _pick(rng, bcfg.get("long_wavelength_sigma_xy", [26.0, 48.0]), float)

    long_wl = _smooth_noise_2d(rng, nx, ny, sigma_xy=sigma_long)
    regional = _smooth_noise_2d(rng, nx, ny, sigma_xy=sigma_long * 1.35)
    relief = 0.65 * long_wl + 0.35 * regional

    n_rel = _pick(rng, bcfg.get("broad_relief_count", [4, 9]), int)
    for _ in range(n_rel):
        cx = rng.uniform(0.08, 0.92)
        cy = rng.uniform(0.08, 0.92)
        rxy = _pick(rng, bcfg.get("broad_relief_radius_xy", [0.12, 0.30]), float)
        rx = rxy * rng.uniform(0.9, 1.4)
        ry = rxy * rng.uniform(0.8, 1.3)
        mag = _pick(rng, bcfg.get("broad_relief_strength", [0.20, 0.55]), float)
        sign = rng.choice([-1.0, 1.0])
        dome = np.exp(-(((xx - cx) / max(rx, 1e-4)) ** 2 + ((yy - cy) / max(ry, 1e-4)) ** 2))
        relief += sign * mag * dome

    top_basement = trend + amp_s * relief
    top_basement = gaussian_filter(top_basement.astype(np.float32), sigma=1.8)

    min_f, max_f = bcfg.get("top_depth_fraction", [0.42, 0.82])
    zmin = float(min_f) * nz
    zmax = float(max_f) * nz

    # Smoothly scale the surface to fit bounds instead of hard-clipping
    current_min, current_max = top_basement.min(), top_basement.max()
    if current_min < zmin or current_max > zmax:
        top_basement = zmin + (top_basement - current_min) * (zmax - zmin) / max((current_max - current_min), 1e-6)

    # Final safety catch just to ensure it stays within grid limits
    top_basement = np.clip(top_basement, 0, nz - 1).astype(np.float32)
    return top_basement



def generate_basement_step1(config_path: str | Path, store: dict | None = None) -> dict[str, str]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dz_m = float(grid["dz_m"])

    out_cfg = cfg.get("output", {})
    out_root = Path(out_cfg.get("output_dir", "./outputs"))
    model_name = str(out_cfg.get("model_name", "model_0000"))
    model_dir = out_root / model_name
    preserve_debug = bool(out_cfg.get("preserve_debug_figures", True))
    if model_dir.exists() and not preserve_debug:
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    top_basement_depth_map = _build_top_basement(cfg, rng)

    z_idx = np.arange(nz, dtype=np.float32)[None, None, :]
    basement_mask = (z_idx >= top_basement_depth_map[:, :, None]).astype(np.uint8)

    weathered_enabled = bool(cfg["basement"].get("weathered_enabled", True))
    weathered_th_m = _pick(rng, cfg["basement"].get("weathered_thickness_m", [8.0, 36.0]), float)
    weathered_th_s = weathered_th_m / max(dz_m, 1e-6)
    if weathered_enabled:
        weathered_mask = (
            (z_idx >= top_basement_depth_map[:, :, None])
            & (z_idx < top_basement_depth_map[:, :, None] + weathered_th_s)
        ).astype(np.uint8)
    else:
        weathered_mask = np.zeros((nx, ny, nz), dtype=np.uint8)

    unit_id = np.zeros((nx, ny, nz), dtype=np.int16)
    facies_id = np.zeros((nx, ny, nz), dtype=np.int16)

    # Dedicated basement IDs.
    unit_id[basement_mask == 1] = 10
    unit_id[weathered_mask == 1] = 11

    facies_id[basement_mask == 1] = 100
    facies_id[weathered_mask == 1] = 101

    basement_top_pick = np.zeros((nx, ny, nz), dtype=np.uint8)
    z_pick = np.clip(np.rint(top_basement_depth_map).astype(np.int32), 0, nz - 1)
    ix = np.arange(nx)[:, None]
    iy = np.arange(ny)[None, :]
    basement_top_pick[ix, iy, z_pick] = 1

    files = {
        "top_basement_depth_map": model_dir / "top_basement_depth_map.npy",
        "basement_mask": model_dir / "basement_mask.npy",
        "weathered_basement_mask": model_dir / "weathered_basement_mask.npy",
        "unit_id": model_dir / "unit_id.npy",
        "facies_id": model_dir / "facies_id.npy",
        "basement_top_pick": model_dir / "basement_top_pick.npy",
    }

    put_array(store, files["top_basement_depth_map"], top_basement_depth_map.astype(np.float32))
    put_array(store, files["basement_mask"], basement_mask.astype(np.uint8))
    put_array(store, files["weathered_basement_mask"], weathered_mask.astype(np.uint8))
    put_array(store, files["unit_id"], unit_id.astype(np.int16))
    put_array(store, files["facies_id"], facies_id.astype(np.int16))
    put_array(store, files["basement_top_pick"], basement_top_pick.astype(np.uint8))

    meta = {
        "step": "step1_basement",
        "seed": seed,
        "domain_type": "geometric_and_categorical_only",
        "grid": {"nx": nx, "ny": ny, "nz": nz, "dz_m": dz_m},
        "weathered_cap_enabled": weathered_enabled,
        "weathered_cap_thickness_m": float(weathered_th_m),
        "unit_id_schema": {"0": "non-basement", "10": "basement_fresh", "11": "basement_weathered"},
        "facies_id_schema": {"0": "none", "100": "basement_fresh", "101": "basement_weathered"},
        "files": {k: str(v) for k, v in files.items()},
        "stats": {
            "top_basement_min_samples": float(top_basement_depth_map.min()),
            "top_basement_max_samples": float(top_basement_depth_map.max()),
            "basement_voxel_fraction": float(basement_mask.mean()),
            "weathered_voxel_fraction": float(weathered_mask.mean()),
        },
    }

    meta_path = model_dir / "metadata_step1.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = {k: str(v) for k, v in files.items()}
    out["metadata_step1"] = str(meta_path)
    return out


def _build_interval_boundary(
    rng: np.random.Generator,
    nx: int,
    ny: int,
    nz: int,
    z_mean: float,
    dz_m: float,
    cfg_int: dict[str, Any]
) -> np.ndarray:
    """Generates a single sub-horizontal boundary depth map."""
    amp_s = _pick(rng, cfg_int.get("relief_amplitude_m", [15.0, 45.0]), float) / max(dz_m, 1e-6)
    sigma_long = _pick(rng, cfg_int.get("long_wavelength_sigma_xy", [25.0, 55.0]), float)

    xx, yy = np.meshgrid(
        np.linspace(0.0, 1.0, nx, dtype=np.float32),
        np.linspace(0.0, 1.0, ny, dtype=np.float32),
        indexing="ij",
    )

    trend_x = _pick(rng, cfg_int.get("trend_x_samples", [-6.0, 6.0]), float)
    trend_y = _pick(rng, cfg_int.get("trend_y_samples", [-6.0, 6.0]), float)

    # Base plane with some random dip
    trend = z_mean + trend_x * (xx - 0.5) + trend_y * (yy - 0.5)

    # Add wavy depositional/erosional relief
    relief = _smooth_noise_2d(rng, nx, ny, sigma_xy=sigma_long)
    boundary = trend + amp_s * relief

    return np.clip(boundary, 0, nz - 1).astype(np.float32)


def generate_intervals_step2(config_path: str | Path, step1_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 2)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dz_m = float(grid["dz_m"])

    # Load required Step 1 outputs
    top_basement = get_array(store, step1_out["top_basement_depth_map"])
    old_weathered_mask = get_array(store, step1_out["weathered_basement_mask"])

    cfg_int = cfg.get("intervals", {})
    configured_intervals = int(cfg_int.get("num_intervals", 3))

    H = []
    F = []

    z_base_mean = float(np.mean(top_basement))
    dz_step = z_base_mean / (configured_intervals + 1)

    # Optional mean-depth jitter per boundary (fraction of dz_step). Nonzero
    # values let adjacent boundaries approach or overlap, deepening erosional
    # truncation (lower preservation ratios). Default 0.0 draws nothing, so
    # legacy configs reproduce the historical rng stream exactly.
    z_jitter_frac = float(cfg_int.get("z_mean_jitter_frac", 0.0))

    # Generate the configured wavy boundaries
    for i in range(1, configured_intervals + 1):
        z_mean = z_base_mean - i * dz_step
        if z_jitter_frac > 0.0:
            z_mean += rng.uniform(-z_jitter_frac, z_jitter_frac) * dz_step
        H_i = _build_interval_boundary(rng, nx, ny, nz, z_mean, dz_m, cfg_int)
        H.append(H_i)
        F.append(np.ones((nx, ny), dtype=bool))

    # --- THE CAP INTERVAL ---
    # Add one final, perfectly flat boundary at Z=0 to fill the remaining space
    H.append(np.zeros((nx, ny), dtype=np.float32))
    F.append(np.ones((nx, ny), dtype=bool))

    total_intervals = configured_intervals + 1

    # --- 1. Track Candidate Geometries ---
    from scipy.ndimage import gaussian_filter

    E = top_basement.copy()
    candidate_top = np.zeros((total_intervals, nx, ny), dtype=np.float32)
    candidate_bottom = np.zeros((total_intervals, nx, ny), dtype=np.float32)

    for i in range(total_intervals):
        H_i = H[i]
        F_i = F[i]

        candidate_bottom[i] = E.copy()
        candidate_top[i] = H_i.copy()
        E = np.where(F_i, H_i, E)

    # --- 2. Resolve Preserved Geometries ---
    preserved_top = candidate_top.copy()
    preserved_bottom = candidate_bottom.copy()
    preserved_basement_top = top_basement.copy()

    for j in range(total_intervals):
        H_j = H[j]
        F_j = F[j]

        preserved_basement_top = np.where(
            F_j,
            np.maximum(preserved_basement_top, H_j),
            preserved_basement_top
        )
        preserved_basement_top = gaussian_filter(preserved_basement_top, sigma=1.0)

        for k in range(j):
            preserved_top[k] = np.where(
                F_j,
                np.maximum(preserved_top[k], H_j),
                preserved_top[k]
            )
            preserved_top[k] = gaussian_filter(preserved_top[k], sigma=1.0)

    # Re-derive each slab's bottom from the resolved top of the slab beneath it.
    #
    # In candidate space consecutive slabs share a surface exactly
    # (candidate_bottom[i] is candidate_top[i-1] by construction). The loop
    # above then truncates and Gaussian-smooths `preserved_top` but leaves
    # `preserved_bottom` at its candidate value, so the one shared surface ends
    # up with two divergent representations. Wherever the smoothed top lands
    # deeper than the untouched bottom, a sub-voxel sliver belongs to neither
    # slab and voxelizes into a thin sheet of unlabeled (unit_id == 0)
    # sediment -- ~4k voxels per 256^3 model, which the DL interval label then
    # maps to class 0 ("basement/eroded") even though the voxels are sediment.
    #
    # Assigning bottoms from tops makes the two representations the same array
    # again, so gaps are impossible regardless of how much smoothing is applied.
    # Set intervals.gap_free_bottoms: false to reproduce the pre-fix behavior.
    if bool(cfg_int.get("gap_free_bottoms", True)):
        preserved_bottom[0] = preserved_basement_top
        for i in range(1, total_intervals):
            preserved_bottom[i] = preserved_top[i - 1]

    for i in range(total_intervals):
        preserved_bottom[i] = np.maximum(preserved_bottom[i], preserved_top[i])

    # --- 3. Build 3D Masks and Volumes ---
    z_idx = np.arange(nz, dtype=np.float32)[None, None, :]

    candidate_masks = np.zeros((total_intervals, nx, ny, nz), dtype=np.uint8)
    preserved_masks = np.zeros((total_intervals, nx, ny, nz), dtype=np.uint8)

    for i in range(total_intervals):
        c_mask = (z_idx >= candidate_top[i, :, :, None]) & (z_idx < candidate_bottom[i, :, :, None])
        candidate_masks[i] = c_mask.astype(np.uint8)

        p_mask = (z_idx >= preserved_top[i, :, :, None]) & (z_idx < preserved_bottom[i, :, :, None])
        preserved_masks[i] = p_mask.astype(np.uint8)

    basement_mask = (z_idx >= preserved_basement_top[:, :, None]).astype(np.uint8)
    weathered_basement_mask = old_weathered_mask & basement_mask

    unit_id_vol = np.zeros((nx, ny, nz), dtype=np.int16)
    facies_id_vol = np.zeros((nx, ny, nz), dtype=np.int16)

    unit_id_vol[basement_mask == 1] = 10
    unit_id_vol[weathered_basement_mask == 1] = 11
    facies_id_vol[basement_mask == 1] = 100
    facies_id_vol[weathered_basement_mask == 1] = 101

    for i in range(total_intervals):
        unit_id = i + 1
        facies_id = (i + 1) * 10
        mask = preserved_masks[i] == 1
        unit_id_vol[mask] = unit_id
        facies_id_vol[mask] = facies_id

    # --- 4. Export Outputs ---
    model_dir = Path(step1_out["top_basement_depth_map"]).parent
    files = {
        "top_basement_depth_map": model_dir / "top_basement_depth_map.npy",
        "basement_mask": model_dir / "basement_mask.npy",
        "weathered_basement_mask": model_dir / "weathered_basement_mask.npy",
        "interval_candidate_top_depth_maps": model_dir / "interval_candidate_top_depth_maps.npy",
        "interval_candidate_bottom_depth_maps": model_dir / "interval_candidate_bottom_depth_maps.npy",
        "interval_preserved_top_depth_maps": model_dir / "interval_preserved_top_depth_maps.npy",
        "interval_preserved_bottom_depth_maps": model_dir / "interval_preserved_bottom_depth_maps.npy",
        "interval_candidate_masks": model_dir / "interval_candidate_masks.npy",
        "interval_resolved_masks": model_dir / "interval_resolved_masks.npy",
        "unit_id_vol": model_dir / "unit_id_vol.npy",
        "facies_id_vol": model_dir / "facies_id_vol.npy",
    }

    put_array(store, files["top_basement_depth_map"], preserved_basement_top)
    put_array(store, files["basement_mask"], basement_mask)
    put_array(store, files["weathered_basement_mask"], weathered_basement_mask)
    put_array(store, files["interval_candidate_top_depth_maps"], candidate_top)
    put_array(store, files["interval_candidate_bottom_depth_maps"], candidate_bottom)
    put_array(store, files["interval_preserved_top_depth_maps"], preserved_top)
    put_array(store, files["interval_preserved_bottom_depth_maps"], preserved_bottom)
    put_array(store, files["interval_candidate_masks"], candidate_masks)
    put_array(store, files["interval_resolved_masks"], preserved_masks)
    put_array(store, files["unit_id_vol"], unit_id_vol)
    put_array(store, files["facies_id_vol"], facies_id_vol)

    meta = {
        "step": "step2_intervals",
        "num_intervals_total": total_intervals,
        "domain_type": "depositional_sequences",
        "unit_id_schema": {"0": "eroded/empty", "1-N": "intervals", "10": "basement", "11": "weathered_basement"},
        "files": {k: str(v) for k, v in files.items()},
    }

    meta_path = model_dir / "metadata_step2.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = {k: str(v) for k, v in files.items()}
    out["metadata_step2"] = str(meta_path)
    return out

def generate_architecture_step3(config_path: str | Path, step2_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 3)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    arch_cfg = cfg.get("architecture", {})
    avg_bed_thick = float(arch_cfg.get("avg_bed_thickness_voxels", 4.0))
    mode = arch_cfg.get("mode", "random")

    # Load required Step 2 outputs
    candidate_tops = get_array(store, step2_out["interval_candidate_top_depth_maps"])
    candidate_bottoms = get_array(store, step2_out["interval_candidate_bottom_depth_maps"])
    preserved_masks = get_array(store, step2_out["interval_resolved_masks"])
    unit_id_vol = get_array(store, step2_out["unit_id_vol"])

    num_intervals = candidate_tops.shape[0]

    # Initialize our new high-resolution bed volume
    # Basement (10) and Weathered (11) remain unlayered for now
    bed_id_vol = unit_id_vol.copy().astype(np.int32)

    # 3D Z-index grid for distance calculations
    z_3d = np.arange(nz, dtype=np.float32).reshape(1, 1, nz)

    metadata_arch = {}

    for i in range(num_intervals):
        # 1. Decide the architecture style for this interval
        styles = ["proportional", "top_parallel", "bottom_parallel"]
        style = rng.choice(styles) if mode == "random" else mode

        # 2. Extract 2D boundaries and convert to 3D for vectorized math
        top_2d = candidate_tops[i]
        bot_2d = candidate_bottoms[i]

        max_thick = np.max(bot_2d - top_2d)
        if max_thick <= 0:
            continue  # Interval was completely eroded everywhere

        top_3d = top_2d[..., None]
        bot_3d = bot_2d[..., None]
        thick_3d = bot_3d - top_3d

        # 3. Generate randomized bed thicknesses
        # We generate enough beds to fill the thickest part of the interval
        num_beds = int(max_thick / avg_bed_thick) + 5
        bed_thicknesses = rng.uniform(avg_bed_thick * 0.4, avg_bed_thick * 1.6, size=num_beds)
        bed_boundaries = np.cumsum(bed_thicknesses)
        bed_boundaries = np.insert(bed_boundaries, 0, 0.0) # Start at 0

        # 4. Calculate the continuous depositional coordinate
        if style == "proportional":
            # Normalize boundaries to [0, 1]
            norm_boundaries = bed_boundaries / bed_boundaries[-1]
            # Protect against division by zero
            safe_thick = np.where(thick_3d < 1e-5, 1e-5, thick_3d)
            coord = (z_3d - top_3d) / safe_thick
            bed_idx = np.digitize(coord, norm_boundaries)

        elif style == "top_parallel":
            # Distance from the top surface
            coord = z_3d - top_3d
            bed_idx = np.digitize(coord, bed_boundaries)

        elif style == "bottom_parallel":
            # Distance from the bottom surface
            coord = bot_3d - z_3d
            bed_idx = np.digitize(coord, bed_boundaries)

        # 5. Narrow-region fix:
        # If the candidate thickness is < 2 voxels, force it to be a single bed (bed 1)
        bed_idx = np.where(thick_3d < 2.0, 1, bed_idx)

        # 6. Assign globally unique Bed IDs
        # E.g., Interval 1 -> Beds 1001, 1002, 1003... Interval 2 -> 2001, 2002...
        base_id = (i + 1) * 1000
        final_bed_ids = base_id + bed_idx

        # 7. Apply the Preserved Mask (The Geological Truncation!)
        mask = preserved_masks[i] == 1
        bed_id_vol[mask] = final_bed_ids[mask]

        metadata_arch[f"interval_{i+1}"] = {
            "style": str(style),
            "max_candidate_thickness": float(max_thick),
            "num_generated_beds": int(num_beds)
        }

    # --- Export Outputs ---
    model_dir = Path(step2_out["unit_id_vol"]).parent
    files = {"bed_id_vol": model_dir / "bed_id_vol.npy"}
    put_array(store, files["bed_id_vol"], bed_id_vol)

    meta = {
        "step": "step3_architecture",
        "domain_type": "internal_layering",
        "interval_metadata": metadata_arch,
        "files": {k: str(v) for k, v in files.items()},
    }

    meta_path = model_dir / "metadata_step3.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # Return combined outputs
    out = step2_out.copy()
    out.update({k: str(v) for k, v in files.items()})
    out["metadata_step3"] = str(meta_path)
    return out

def generate_geobodies_step4(config_path: str | Path, step3_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 4)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    ch_cfg = cfg.get("geobodies", {}).get("channels", {})
    num_channels = int(ch_cfg.get("num_channels", 3))
    # Channel flow orientation:
    #   "x"      — legacy behavior, every channel flows along the inline (x) axis
    #   "random" — each channel flows along x or y with equal probability,
    #              removing the systematic orientation anisotropy of the
    #              legacy dataset. Adds one rng draw per channel, so datasets
    #              generated with "random" are not reproductions of legacy runs.
    orientation = str(ch_cfg.get("orientation", "x"))

    # Load required volumes from previous steps
    bed_id_vol = get_array(store, step3_out["bed_id_vol"])
    facies_id_vol = get_array(store, step3_out["facies_id_vol"])
    preserved_tops = get_array(store, step3_out["interval_preserved_top_depth_maps"])
    preserved_masks = get_array(store, step3_out["interval_resolved_masks"])

    num_intervals = preserved_tops.shape[0]

    # Create a dedicated mask for channels (0 = background, 1-N = channel IDs)
    channel_mask_vol = np.zeros((nx, ny, nz), dtype=np.int16)

    # Pre-compute 3D Z grid for math
    z_3d = np.arange(nz, dtype=np.float32).reshape(1, 1, nz)
    x_idx = np.arange(nx)
    y_idx = np.arange(ny)
    xx, yy = np.meshgrid(x_idx, y_idx, indexing="ij")

    metadata_channels = {}

    for i in range(num_channels):
        channel_id = i + 1
        # 1. Randomly pick a host interval (excluding the absolute top cap if possible)
        # Using rng.integers(0, num_intervals - 1) to avoid the empty air cap
        host_int = rng.integers(0, max(1, num_intervals - 1))

        # 2. Pick channel parameters
        W = rng.uniform(*ch_cfg.get("width_voxels", [10.0, 30.0]))
        D = rng.uniform(*ch_cfg.get("depth_voxels", [5.0, 15.0]))
        freq = rng.uniform(*ch_cfg.get("sinuosity_freq", [0.5, 1.5]))
        amp = rng.uniform(*ch_cfg.get("amplitude_voxels", [10.0, 40.0]))

        # 2b. Flow orientation (extra draw only in "random" mode, so legacy
        #     configs reproduce the historical rng stream exactly)
        flow_along_y = (orientation == "random") and bool(rng.random() < 0.5)

        # 3. Generate meandering 2D centerline
        if flow_along_y:
            # Flowing roughly along Y: centerline is x(y)
            x0 = rng.uniform(0.2 * nx, 0.8 * nx)
            center_x = x0 + amp * np.sin(2 * np.pi * freq * (yy / ny))
            dist_to_center = np.abs(xx - center_x)
        else:
            # Flowing roughly along X: centerline is y(x) (legacy behavior)
            y0 = rng.uniform(0.2 * ny, 0.8 * ny)
            center_y = y0 + amp * np.sin(2 * np.pi * freq * (xx / nx))
            dist_to_center = np.abs(yy - center_y)

        # 4. Calculate U-shaped depth profile (Parabola)
        # Depth is max (D) at center, 0 at edges (W/2)
        scour_depth = np.where(
            dist_to_center < W / 2.0,
            D * (1.0 - (dist_to_center / (W / 2.0))**2),
            0.0
        )

        # 5. Hang the channel from the host interval's preserved top
        host_top = preserved_tops[host_int]
        channel_base = host_top + scour_depth  # Z increases downwards!

        # 6. Create 3D geometric mask
        c_mask_3d = (z_3d >= host_top[..., None]) & (z_3d < channel_base[..., None])

        # RULE ENFORCEMENT: Clip by the host interval mask.
        # This prevents the channel from entering older intervals or the basement.
        host_mask_3d = (preserved_masks[host_int] == 1)
        final_mask = c_mask_3d & host_mask_3d

        # 7. Internal Architecture (Bottom-controlled)
        # Calculate distance from the parabolic base upwards
        dist_from_base = channel_base[..., None] - z_3d
        bed_thick = float(ch_cfg.get("internal_bed_thick_voxels", 2.0))

        # Channels get base IDs starting at 50000, 51000, etc.
        base_ch_id = (channel_id * 1000) + 50000
        ch_bed_idx = np.floor(dist_from_base / bed_thick).astype(np.int32)
        final_ch_beds = base_ch_id + ch_bed_idx

        # 8. Override the host properties!
        channel_mask_vol[final_mask] = channel_id
        bed_id_vol[final_mask] = final_ch_beds[final_mask]
        facies_id_vol[final_mask] = 50  # 50 = standard channel sand facies

        metadata_channels[f"channel_{channel_id}"] = {
            "host_interval_id": int(host_int + 1),
            "flow_axis": "y" if flow_along_y else "x",
            "width_voxels": float(W),
            "max_depth_voxels": float(D)
        }

    # --- Export Outputs ---
    model_dir = Path(step3_out["bed_id_vol"]).parent
    files = {
        "channel_mask_vol": model_dir / "channel_mask_vol.npy",
        "final_bed_id_vol": model_dir / "final_bed_id_vol.npy",
        "final_facies_id_vol": model_dir / "final_facies_id_vol.npy"
    }

    put_array(store, files["channel_mask_vol"], channel_mask_vol)
    put_array(store, files["final_bed_id_vol"], bed_id_vol)
    put_array(store, files["final_facies_id_vol"], facies_id_vol)

    meta = {
        "step": "step4_geobodies",
        "channels_injected": num_channels,
        "channel_orientation_mode": orientation,
        "channel_metadata": metadata_channels,
        "facies_schema": {"50": "channel_sand"},
        "files": {k: str(v) for k, v in files.items()},
    }

    meta_path = model_dir / "metadata_step4.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = step3_out.copy()
    out.update({k: str(v) for k, v in files.items()})
    out["metadata_step4"] = str(meta_path)
    return out

def generate_structural_deformation_step5(config_path: str | Path, step4_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    from scipy.ndimage import map_coordinates
    import json

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 5)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    def_cfg = cfg.get("deformation", {}).get("faults", {})
    num_faults = int(def_cfg.get("num_faults", 3))
    prob_blind = float(def_cfg.get("blind_fault_probability", 0.5))
    min_spacing = float(def_cfg.get("min_spacing_voxels", 40.0))
    # Fault kinematics:
    #   "normal" — the hanging wall (the block above the plane) slips down-dip,
    #              parallel to the fault plane: true normal faults.
    #   "legacy" — pre-fix behavior, kept for reproducing earlier datasets.
    #              It translates the block BELOW the plane (the
    #              footwall) downward along a vector that is not parallel to
    #              the plane, which produces reverse-sense offsets (repeated
    #              section in vertical columns) instead of normal faults.
    # No rng draws depend on this switch, so fault positions are unchanged.
    kinematics = str(def_cfg.get("kinematics", "normal"))
    # Blind-fault taper length, as a multiple of the fault's displacement
    # (normal kinematics only). The taper runs over at least this depth range
    # even when the tip interval is thinner. Without it a thin tip interval
    # takes the whole lateral slip component (cos(dip) * disp) across a few
    # voxels: shear strains > 1 that overturn dipping contacts (younger strata
    # beneath older) and tip displacement gradients ~1, far above the
    # ~0.1-0.3 observed on natural faults. With 2.0 the gradient stays below
    # ~0.3 and the shear strain below ~0.23.
    taper_min_disp = float(def_cfg.get("blind_taper_min_length_disp", 2.0))
    if kinematics not in ("normal", "legacy"):
        raise ValueError(f"deformation.faults.kinematics must be 'normal' or 'legacy', got {kinematics!r}")

    bed_id_vol = get_array(store, step4_out["final_bed_id_vol"])
    facies_id_vol = get_array(store, step4_out["final_facies_id_vol"])
    channel_mask_vol = get_array(store, step4_out["channel_mask_vol"])
    top_basement = get_array(store, step4_out["top_basement_depth_map"])
    preserved_tops = get_array(store, step4_out["interval_preserved_top_depth_maps"])

    num_intervals = preserved_tops.shape[0]

    fault_plane_vol = np.zeros((nx, ny, nz), dtype=np.uint8)

    # PERF: broadcastable 1-D index axes replace the three full int64 meshgrid
    # volumes (3 x N^3 x 8 bytes) the legacy implementation materialized.
    # Broadcasting produces elementwise-identical float64 results, without the
    # multi-GB allocations at 512^3.
    x_bc = np.arange(nx, dtype=np.int64).reshape(-1, 1, 1)
    y_bc = np.arange(ny, dtype=np.int64).reshape(1, -1, 1)
    z_bc = np.arange(nz, dtype=np.int64).reshape(1, 1, -1)

    # PERF: a flat source-index volume lets us run scipy's order-0 coordinate
    # mapping ONCE per fault and reuse the mapped indices as a cheap gather for
    # every deformed volume. Order-0 (nearest-neighbor) mapping just selects a
    # source voxel per output voxel, so gathering any volume by the mapped
    # index is bit-identical to mapping that volume directly.
    src_flat_idx = np.arange(nx * ny * nz, dtype=np.int32).reshape(nx, ny, nz)

    metadata_faults = {}

    # Global mask to track the 3D "forcefields" of all accepted faults
    existing_planes_mask = np.zeros((nx, ny, nz), dtype=bool)

    for f in range(num_faults):
        valid_fault = False
        max_attempts = 100

        # --- REJECTION SAMPLING (FULL 3D AVOIDANCE) ---
        for attempt in range(max_attempts):
            disp = rng.uniform(*def_cfg.get("max_displacement_voxels", [10.0, 25.0]))
            dip = np.radians(rng.uniform(*def_cfg.get("dip_degrees", [50.0, 75.0])))
            strike = np.radians(rng.uniform(*def_cfg.get("strike_degrees", [0.0, 360.0])))

            cx = rng.uniform(0.2 * nx, 0.8 * nx)
            cy = rng.uniform(0.2 * ny, 0.8 * ny)
            cz = rng.uniform(0.3 * nz, 0.7 * nz)

            nx_n = np.sin(dip) * np.cos(strike)
            ny_n = np.sin(dip) * np.sin(strike)
            nz_n = np.cos(dip)

            # Calculate distance of every voxel to this candidate plane
            dist_to_plane = (x_bc - cx) * nx_n + (y_bc - cy) * ny_n + (z_bc - cz) * nz_n

            # Create a "fat" safety buffer around this candidate plane
            candidate_fat_plane = np.abs(dist_to_plane) < (min_spacing / 2.0)

            # Check if this new plane intersects ANY existing plane inside our 3D grid!
            if not np.any(candidate_fat_plane & existing_planes_mask):
                valid_fault = True
                # Add this plane's forcefield to the global tracker
                existing_planes_mask |= candidate_fat_plane
                break

        if not valid_fault:
            print(f"Warning: Grid too crowded! Skipping Fault {f+1} entirely.")
            continue  # <--- This tells Python to abort this fault and move on
        # ---------------------------------------------------------

        # The plane normal (nx_n, ny_n, nz_n) points down (nz_n > 0, z is depth)
        # and toward +(cos strike, sin strike); the plane therefore dips toward
        # -(cos strike, sin strike). Points with dist_to_plane < 0 lie above
        # the plane: the hanging wall.
        if kinematics == "legacy":
            hw_mask = dist_to_plane > 0
            sx = np.cos(dip) * np.cos(strike)
            sy = np.cos(dip) * np.sin(strike)
            sz = np.sin(dip)
        else:
            hw_mask = dist_to_plane < 0
            # Unit down-dip vector lying IN the plane (s . n = 0): the hanging
            # wall moves down and toward the dip direction -> normal fault.
            sx = -np.cos(dip) * np.cos(strike)
            sy = -np.cos(dip) * np.sin(strike)
            sz = np.sin(dip)

        inv_sx = -sx * disp
        inv_sy = -sy * disp
        inv_sz = -sz * disp

        z_decay = np.ones((nx, ny, nz), dtype=np.float32)
        is_blind = rng.random() < prob_blind

        if is_blind and num_intervals > 2:
            tip_int = rng.integers(1, num_intervals - 1)
            tip_top = preserved_tops[tip_int][:, :, None]
            tip_base = preserved_tops[tip_int - 1][:, :, None] if tip_int > 0 else np.zeros_like(tip_top)

            int_thickness = tip_base - tip_top
            int_thickness = np.where(int_thickness < 1, 1, int_thickness)
            if kinematics == "normal":
                # The taper is meant to grow slip from zero at the tip
                # interval's top to full at its base in the PRE-deformation
                # geometry. Evaluated at output depth over the bare thickness T
                # (legacy), the inverse map z -> z - a*decay(z) has gradient
                # 1 - a/T and folds -- duplicating strata -- wherever the
                # vertical throw a = sz*disp exceeds T. Solving the Lagrangian
                # taper exactly for the output depth gives the same linear ramp
                # over T + a, whose gradient T/(T + a) is always positive.
                # T is floored at taper_min_disp * disp (see above).
                int_thickness = np.maximum(int_thickness, taper_min_disp * disp) + sz * disp

            decay_zone = (z_bc - tip_top) / int_thickness
            z_decay = np.clip(decay_zone, 0.0, 1.0)
            z_decay[np.broadcast_to(z_bc, z_decay.shape) < np.broadcast_to(tip_top, z_decay.shape)] = 0.0

        shift_x = np.where(hw_mask, inv_sx * z_decay, 0.0)
        shift_y = np.where(hw_mask, inv_sy * z_decay, 0.0)
        shift_z = np.where(hw_mask, inv_sz * z_decay, 0.0)

        map_coords = [x_bc + shift_x, y_bc + shift_y, z_bc + shift_z]

        # One order-0 mapping of the flat index volume, then cheap gathers for
        # all four deformed volumes (see src_flat_idx comment above).
        mapped_idx = map_coordinates(src_flat_idx, map_coords, order=0, mode='nearest')
        bed_id_vol = bed_id_vol.reshape(-1)[mapped_idx]
        facies_id_vol = facies_id_vol.reshape(-1)[mapped_idx]
        channel_mask_vol = channel_mask_vol.reshape(-1)[mapped_idx]
        fault_plane_vol = fault_plane_vol.reshape(-1)[mapped_idx]
        del mapped_idx, map_coords

        plane_mask = np.abs(dist_to_plane) < 1.0
        above_basement = (bed_id_vol != 10) & (bed_id_vol != 11)
        active_slip = z_decay > 0.05

        current_fault_mask = (plane_mask) & (above_basement) & (active_slip)
        fault_plane_vol[current_fault_mask] = 1

        for i in range(num_intervals):
            z_map = preserved_tops[i]
            z_map_int = np.clip(np.round(z_map).astype(int), 0, nz - 1)
            x_2d, y_2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            local_shift_z = shift_z[x_2d, y_2d, z_map_int]
            preserved_tops[i] = z_map - local_shift_z

        # Apply the exact same 2D displacement to the Basement Top!
        z_base_int = np.clip(np.round(top_basement).astype(int), 0, nz - 1)
        local_shift_z_base = shift_z[x_2d, y_2d, z_base_int]
        top_basement = top_basement - local_shift_z_base

        metadata_faults[f"fault_{f+1}"] = {
            "type": "blind" if is_blind else "post_depositional",
            "displacement_voxels": float(disp),
            "dip_degrees": float(np.degrees(dip)),
            # Full plane geometry, so a fault can be reconstructed exactly
            # from metadata (recording it consumes no rng draws).
            "strike_degrees": float(np.degrees(strike)),
            "anchor_voxels": [float(cx), float(cy), float(cz)],
            "tip_interval_id": int(tip_int + 1) if (is_blind and num_intervals > 2) else None,
            "sense": "normal" if kinematics == "normal" else "legacy",
        }

    # --- Export Outputs ---
    model_dir = Path(step4_out["final_bed_id_vol"]).parent
    files = {
        "faulted_bed_id_vol": model_dir / "faulted_bed_id_vol.npy",
        "faulted_facies_id_vol": model_dir / "faulted_facies_id_vol.npy",
        "fault_plane_vol": model_dir / "fault_plane_vol.npy",
        "faulted_preserved_tops": model_dir / "faulted_preserved_tops.npy",
        "faulted_top_basement": model_dir / "faulted_top_basement.npy",
    }

    put_array(store, files["faulted_bed_id_vol"], bed_id_vol)
    put_array(store, files["faulted_facies_id_vol"], facies_id_vol)
    put_array(store, files["fault_plane_vol"], fault_plane_vol)
    put_array(store, files["faulted_preserved_tops"], preserved_tops)
    put_array(store, files["faulted_top_basement"], top_basement)

    meta = {
        "step": "step5_deformation",
        "faults_generated": num_faults,
        "fault_kinematics": kinematics,
        "fault_metadata": metadata_faults,
        "files": {k: str(v) for k, v in files.items()},
    }

    meta_path = model_dir / "metadata_step5.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = step4_out.copy()
    out.update({k: str(v) for k, v in files.items()})
    out["metadata_step5"] = str(meta_path)
    return out

def generate_rock_properties_step6(config_path: str | Path, step5_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    dz_m = float(grid["dz_m"])

    prop_cfg = cfg.get("properties", {})

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 6)

    # Randomized per-model basement offset above the local depth-compaction trend.
    # Keeping this moderate ensures basement is not the dominant reflector.
    basement_vp_offset = _pick(rng, prop_cfg.get("basement_vp_offset", [300.0, 600.0]), float)
    basement_rho_offset = _pick(rng, prop_cfg.get("basement_rho_offset", [0.10, 0.20]), float)

    # Load the faulted geometry
    bed_id_vol = get_array(store, step5_out["faulted_bed_id_vol"])

    # Initialize physical volumes
    vp_vol = np.zeros((nx, ny, nz), dtype=np.float32)
    rho_vol = np.zeros((nx, ny, nz), dtype=np.float32)

    # PERF: the depth-compaction trends depend only on z, so they are 1-D
    # lookup lines of length nz instead of the full (nx,ny,nz) meshgrid
    # volumes the legacy implementation built (which also allocated two unused
    # full int64 x/y index volumes). Elementwise math on the same float32 z
    # values produces bit-identical trend values.
    z_line = np.arange(nz, dtype=np.float32)
    depth_m_line = z_line * dz_m
    vp_trend_line = prop_cfg.get("sediment_base_vp", 2200.0) + (depth_m_line * prop_cfg.get("vp_depth_gradient", 0.6))
    rho_trend_line = prop_cfg.get("sediment_base_rho", 2.0) + (depth_m_line * prop_cfg.get("rho_depth_gradient", 0.0002))

    # --- Pre-compute weathered-zone fraction (0 at top, 1 at bottom) per voxel ---
    # GEOLOGICAL NOTE: A real weathered basement is a GRADATIONAL profile
    # whose properties transition smoothly from sediment-like at its top to
    # fresh-basement-like at its bottom. Giving the whole weathered zone
    # constant properties would create TWO sharp impedance contrasts
    # (sediment->weathered, weathered->fresh) separated by only 2-7 voxels;
    # the low-frequency wavelet sees both as a single tuned reflector and
    # constructively stacks them into a saturated bright basement event.
    # Spreading the contrast across the weathered thickness produces several
    # small reflections that don't tune-stack, which is geologically realistic.
    weathered_mask_3d = (bed_id_vol == 11)
    if weathered_mask_3d.any():
        first_w_idx = np.argmax(weathered_mask_3d, axis=2).astype(np.float32)
        last_w_idx = (nz - 1 - np.argmax(weathered_mask_3d[:, :, ::-1], axis=2)).astype(np.float32)
        first_3d = first_w_idx[:, :, None]
        last_3d = last_w_idx[:, :, None]
        thickness_3d = np.clip(last_3d - first_3d, 1.0, None)
        z_idx_3d_w = np.arange(nz, dtype=np.float32).reshape(1, 1, nz)
        weathered_fraction = np.clip((z_idx_3d_w - first_3d) / thickness_3d, 0.0, 1.0).astype(np.float32)
        weathered_fraction_flat = weathered_fraction.reshape(-1)
    else:
        weathered_fraction_flat = None

    # PERF: the legacy implementation ran `bed_id_vol == bed` — a full-volume
    # scan — once per unique bed id (hundreds of scans at 512^3). A single
    # stable argsort groups the voxel indices of every bed at once; stable
    # ordering keeps each bed's voxels in C order, exactly matching the order
    # a boolean mask would enumerate them, so per-voxel random draws land on
    # the same voxels in the same order as before.
    flat_bed = bed_id_vol.reshape(-1)
    sort_order = np.argsort(flat_bed, kind="stable")
    sorted_beds = flat_bed[sort_order]
    group_starts = np.flatnonzero(np.r_[True, sorted_beds[1:] != sorted_beds[:-1]])
    group_ends = np.r_[group_starts[1:], sorted_beds.size]
    unique_beds = sorted_beds[group_starts]

    vp_flat = vp_vol.reshape(-1)
    rho_flat = rho_vol.reshape(-1)

    for g in range(unique_beds.size):
        bed = unique_beds[g]
        idx = sort_order[group_starts[g]:group_ends[g]]
        n_vox = idx.size
        zi = idx % nz  # z coordinate of each voxel in the bed

        # Localized generator just for this bed ID (identical to legacy)
        bed_rng = np.random.default_rng(int(bed))

        if bed == 0:
            # Empty / Air / Water
            vp_flat[idx] = prop_cfg.get("water_vp", 1500.0)
            rho_flat[idx] = prop_cfg.get("water_rho", 1.0)

        elif bed == 10:
            # Fresh basement: same compaction trend as deep sediments + moderate offset.
            # Architecturally distinct (chaotic internal noise, no layering) but not
            # anomalously fast — prevents the basement from dominating the section.
            noise_vp = bed_rng.normal(0, 30, size=n_vox)
            noise_rho = bed_rng.normal(0, 0.01, size=n_vox)
            vp_flat[idx] = vp_trend_line[zi] + basement_vp_offset + noise_vp
            rho_flat[idx] = rho_trend_line[zi] + basement_rho_offset + noise_rho

        elif bed == 11:
            # Gradational weathered basement: ramps from pure sediment properties
            # at the top (f=0) to the same offset used by fresh basement at the
            # bottom (f=1), so the weathered-to-fresh boundary has zero extra contrast.
            f = weathered_fraction_flat[idx]
            noise_vp = bed_rng.normal(0, 100, size=n_vox)
            noise_rho = bed_rng.normal(0, 0.04, size=n_vox)
            vp_flat[idx] = vp_trend_line[zi] + f * basement_vp_offset + noise_vp
            rho_flat[idx] = rho_trend_line[zi] + f * basement_rho_offset + noise_rho

        elif bed >= 50000:
            # Channels
            vp_shift = bed_rng.uniform(-200, 200)
            rho_shift = bed_rng.uniform(-0.1, 0.1)
            vp_flat[idx] = prop_cfg.get("channel_vp", 3200.0) + (vp_trend_line[zi] * 0.1) + vp_shift
            rho_flat[idx] = prop_cfg.get("channel_rho", 2.2) + rho_shift

        else:
            # Background Sedimentary Beds — DEPTH-DRIVEN COMPACTION & LITHIFICATION
            # The mean z of each bed (not its interval index) drives compaction
            # and lithification, keeping the behavior independent of bed-ID
            # numbering conventions.
            mean_z = float(np.mean(z_line[zi]))
            mean_z_norm = mean_z / max(nz, 1)   # 0 at surface, 1 at bottom of grid

            # Lithification: deep beds converge in properties (less inter-bed
            # variability). Shallow unconsolidated beds keep their full contrast.
            lithification_factor = 1.0 - 0.6 * mean_z_norm   # 1.0 (top) → 0.4 (bottom)

            # Extra compaction beyond the linear depth gradient (e.g., from
            # diagenetic cementation that grows non-linearly with burial).
            extra_compaction_vp = mean_z_norm * 200.0
            extra_compaction_rho = mean_z_norm * 0.04

            # Per-bed property variability (scaled by lithification)
            vp_bed_var = bed_rng.uniform(-350, 350) * lithification_factor
            rho_bed_var = bed_rng.uniform(-0.15, 0.15) * lithification_factor

            vp_flat[idx] = vp_trend_line[zi] + extra_compaction_vp + vp_bed_var
            rho_flat[idx] = rho_trend_line[zi] + extra_compaction_rho + rho_bed_var

    # Calculate Acoustic Impedance (AI = Vp * Density)
    ai_vol = vp_vol * rho_vol

    # --- Export Outputs ---
    model_dir = Path(step5_out["faulted_bed_id_vol"]).parent
    files = {
        "vp_vol": model_dir / "vp_vol.npy",
        "rho_vol": model_dir / "rho_vol.npy",
        "ai_vol": model_dir / "ai_vol.npy"
    }

    put_array(store, files["vp_vol"], vp_vol)
    put_array(store, files["rho_vol"], rho_vol)
    put_array(store, files["ai_vol"], ai_vol)

    meta = {
        "step": "step6_properties",
        "domain_type": "physical_properties",
        "files": {k: str(v) for k, v in files.items()}
    }

    meta_path = model_dir / "metadata_step6.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = step5_out.copy()
    out.update({k: str(v) for k, v in files.items()})
    out["metadata_step6"] = str(meta_path)
    return out

def _ricker_spatial(width_voxels: float, length: int) -> np.ndarray:
    """Generates a 1D spatial Ricker wavelet."""
    t = np.arange(-length // 2 + 1, length // 2 + 1)
    f = 1.0 / width_voxels
    pi2 = (np.pi ** 2)
    r = (1.0 - 2.0 * pi2 * (f ** 2) * (t ** 2)) * np.exp(-pi2 * (f ** 2) * (t ** 2))
    return r.astype(np.float32)


def _normalize_rms(vol: np.ndarray) -> np.ndarray:
    """
    Normalize a volume to unit RMS so frequency bands can be blended without
    inheriting the per-band wavelet energy bias. A wider (lower-frequency)
    Ricker wavelet has a larger L2 norm than a narrow one, so convolving the
    same reflectivity series with three wavelets produces three results with
    very different total energy. Without this normalization, the band blending
    inadvertently boosts whichever band has the widest wavelet -- which in our
    Q-factor scheme is the low-frequency band that dominates the deep section,
    creating a false amplitude-increase-with-depth artifact.
    """
    rms = float(np.sqrt(np.mean(vol ** 2)))
    if rms < 1e-8:
        return vol
    return (vol / rms).astype(np.float32)

def generate_seismic_step7(config_path: str | Path, step6_out: dict[str, str], store: dict | None = None) -> dict[str, str]:
    from scipy.ndimage import convolve1d, gaussian_filter
    import json

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed + 7)

    grid = cfg["grid"]
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    seis_cfg = cfg.get("seismic", {})
    noise_level = float(seis_cfg.get("noise_level", 0.15))
    w_len = int(seis_cfg.get("wavelet_length_samples", 81))
    widths = seis_cfg.get("wavelet_widths_voxels", [3.0, 8.0, 20.0])
    lateral_footprint = float(seis_cfg.get("lateral_footprint_voxels", 1.5))

    ai_vol = get_array(store, step6_out["ai_vol"])

    # 1. Calculate Reflectivity
    reflectivity = np.zeros((nx, ny, nz), dtype=np.float32)
    ai_down = ai_vol[:, :, 1:]
    ai_up = ai_vol[:, :, :-1]
    reflectivity[:, :, 1:] = (ai_down - ai_up) / (ai_down + ai_up + 1e-8)

    # 2. Generate Three Wavelets
    w_high = _ricker_spatial(widths[0], w_len)
    w_mid = _ricker_spatial(widths[1], w_len)
    w_low = _ricker_spatial(widths[2], w_len)

    # 3. Vertical Convolution (1D) + RMS-normalize each band
    # CRITICAL: Without RMS normalization, the wider low-frequency wavelet
    # contributes more energy than the narrow high-frequency one. When blended
    # with depth-dependent weights, this causes the deep section (low-freq) to
    # appear brighter than the shallow (high-freq) -- the opposite of real Q.
    seis_high = _normalize_rms(convolve1d(reflectivity, w_high, axis=2, mode='constant'))
    seis_mid  = _normalize_rms(convolve1d(reflectivity, w_mid,  axis=2, mode='constant'))
    seis_low  = _normalize_rms(convolve1d(reflectivity, w_low,  axis=2, mode='constant'))

    # 4. AGGRESSIVE Non-Linear Attenuation (Q-Factor)
    z_idx = np.arange(nz, dtype=np.float32).reshape(1, 1, nz)
    z_norm = z_idx / nz

    # High frequency dies off exponentially in the shallow section
    weight_high = np.exp(-6.0 * z_norm)
    # Low frequency takes over aggressively at the bottom
    weight_low = z_norm ** 1.5
    # Mid fills the gap, normalize to prevent amplitude blowouts
    weight_mid = np.clip(1.0 - weight_high - weight_low, 0.0, 1.0)

    # Normalize weights so they sum to 1.0 everywhere
    total_weight = weight_high + weight_mid + weight_low
    weight_high /= total_weight
    weight_mid /= total_weight
    weight_low /= total_weight

    clean_seismic = (seis_high * weight_high) + (seis_mid * weight_mid) + (seis_low * weight_low)

    # Depth-dependent amplitude taper: shallow reflections stay bright, deep ones fade.
    # exp(-2.0 * z_norm) gives ~1.0 at surface, ~0.37 at 50%, ~0.22 at 75% depth.
    taper_factor = float(seis_cfg.get("amplitude_taper_factor", 2.0))
    clean_seismic *= np.exp(-taper_factor * z_norm)

    # 5. LATERAL SMOOTHING (The Migration Footprint)
    # Blurs heavily in X and Y, but leaves Z (depth) perfectly sharp
    clean_seismic = gaussian_filter(clean_seismic, sigma=(lateral_footprint, lateral_footprint, 0.0))

    # 6. STRUCTURED / COLORED NOISE (with matching Q-factor weighting)
    raw_noise = rng.normal(0, 1, size=(nx, ny, nz)).astype(np.float32)

    # Stretch the noise horizontally (Sigma X=3, Y=3) and squish it vertically (Z=0.5)
    # This creates the "wormy" look of real seismic background noise
    structured_noise = gaussian_filter(raw_noise, sigma=(3.0, 3.0, 0.5))

    # Convolve noise with ALL THREE wavelets and blend using the SAME weights as
    # the signal. If we only used w_low, noise inherits low-freq energy uniformly
    # across depth, which adds spurious bright "deep earth ringing" everywhere
    # and contributes to the apparent deep-amplitude boost.
    noise_high = _normalize_rms(convolve1d(structured_noise, w_high, axis=2, mode='constant'))
    noise_mid  = _normalize_rms(convolve1d(structured_noise, w_mid,  axis=2, mode='constant'))
    noise_low  = _normalize_rms(convolve1d(structured_noise, w_low,  axis=2, mode='constant'))
    band_limited_noise = (noise_high * weight_high) + (noise_mid * weight_mid) + (noise_low * weight_low)

    signal_rms = np.sqrt(np.mean(clean_seismic**2))
    noise_rms = np.sqrt(np.mean(band_limited_noise**2)) + 1e-8

    scaled_noise = band_limited_noise * (signal_rms / noise_rms) * noise_level
    final_seismic = clean_seismic + scaled_noise

    # --- Export Outputs ---
    model_dir = Path(step6_out["ai_vol"]).parent
    files = {
        "reflectivity_vol": model_dir / "reflectivity_vol.npy",
        "seismic_vol": model_dir / "seismic_vol.npy"
    }

    put_array(store, files["reflectivity_vol"], reflectivity)
    put_array(store, files["seismic_vol"], final_seismic)

    meta = {
        "step": "step7_seismic",
        "domain_type": "depth_domain_seismic",
        "stats": {
            "seismic_min": float(final_seismic.min()),
            "seismic_max": float(final_seismic.max()),
            "signal_rms": float(signal_rms)
        },
        "files": {k: str(v) for k, v in files.items()}
    }

    meta_path = model_dir / "metadata_step7.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    out = step6_out.copy()
    out.update({k: str(v) for k, v in files.items()})
    out["metadata_step7"] = str(meta_path)
    return out

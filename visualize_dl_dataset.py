import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

def visualize_dl_model(model_dir_str="outputs/dataset/model_0000", slice_index=128, axis="y"):
    """Plot the vertical section at `axis` = `slice_index` (axis "x" or "y")."""
    out_dir = Path(model_dir_str)
    
    print(f"Loading Deep Learning dataset from: {out_dir}...")
    
    # Safely try to load the DL volumes
    try:
        seis_vol = np.load(out_dir / "DL_Input_Seismic.npy")
        fault_vol = np.load(out_dir / "DL_Label_Faults.npy")
        chan_vol = np.load(out_dir / "DL_Label_Channels.npy")
        base_vol = np.load(out_dir / "DL_Label_Basement.npy")
        int_vol = np.load(out_dir / "DL_Label_Intervals.npy")
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. Ensure the path is correct.\n{e}")
        return

    nx, ny, nz = seis_vol.shape
    
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    n_axis = nx if axis == "x" else ny

    # If the requested slice is out of bounds, default to the middle
    if slice_index >= n_axis or slice_index < 0:
        slice_index = n_axis // 2

    print(f"Extracting 2D slices at {axis.upper()} = {slice_index}...")

    # Extract 2D slices (Transpose so Z is depth)
    take = (lambda v: v[slice_index, :, :].T) if axis == "x" else (lambda v: v[:, slice_index, :].T)
    s_seis = take(seis_vol)
    s_fault = take(fault_vol)
    s_chan = take(chan_vol)
    s_base = take(base_vol)
    s_int = take(int_vol)
    
    # Set up the 1x5 Figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 25))
    
    # Calculate clipping for high-contrast seismic (98th percentile)
    vmax_seis = np.percentile(np.abs(s_seis), 98)
    
    # --- Panel 1: Pure Seismic Input ---
    im1 = axes[0].imshow(s_seis, cmap='RdBu', aspect='auto', interpolation='bilinear', vmin=-vmax_seis, vmax=vmax_seis)
    axes[0].set_title("Input: Raw Seismic", fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label="Amplitude", fraction=0.046, pad=0.04)
    
    # --- Panel 2: Fault Label Overlay ---
    axes[1].imshow(s_seis, cmap='RdBu', aspect='auto', interpolation='bilinear', alpha=0.6, vmin=-vmax_seis, vmax=vmax_seis)
    fault_cmap = ListedColormap(['none', '#00ff00']) # Bright Green (as requested in your snippet)
    axes[1].imshow(s_fault, cmap=fault_cmap, aspect='auto', interpolation='nearest')
    axes[1].set_title("Label: Faults", fontsize=14, fontweight='bold')
    
    # --- Panel 3: Channel Label Overlay ---
    axes[2].imshow(s_seis, cmap='RdBu', aspect='auto', interpolation='bilinear', alpha=0.6, vmin=-vmax_seis, vmax=vmax_seis)
    chan_cmap = ListedColormap(['none', '#ff00ff']) # Switched to Magenta so it stands out against Red/Blue
    axes[2].imshow(s_chan, cmap=chan_cmap, aspect='auto', interpolation='nearest')
    axes[2].set_title("Label: Channels", fontsize=14, fontweight='bold')
    
    # --- Panel 4: Basement Label Overlay ---
    axes[3].imshow(s_seis, cmap='RdBu', aspect='auto', interpolation='bilinear', alpha=0.6, vmin=-vmax_seis, vmax=vmax_seis)
    base_cmap = ListedColormap(['none', 'cyan']) 
    axes[3].imshow(s_base, cmap=base_cmap, aspect='auto', interpolation='nearest', alpha=0.5)
    axes[3].set_title("Label: Basement", fontsize=14, fontweight='bold')
    
    # --- Panel 5: Interval Stratigraphy Overlay ---
    np.random.seed(42) 
    max_interval = int(np.max(s_int))
    int_colors = ['none'] 
    for _ in range(max_interval):
        int_colors.append(tuple(np.random.uniform(0.3, 0.9, 3))) # Pastel RGB
    int_cmap = ListedColormap(int_colors)
    
    axes[4].imshow(s_seis, cmap='RdBu', aspect='auto', interpolation='bilinear', alpha=0.4, vmin=-vmax_seis, vmax=vmax_seis)
    axes[4].imshow(s_int, cmap=int_cmap, aspect='auto', interpolation='nearest', alpha=0.8)
    axes[4].set_title("Label: Intervals", fontsize=14, fontweight='bold')
    
    # Clean up axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Y" if axis == "x" else "X", fontsize=10)
        ax.set_ylabel(f"Depth (Z)", fontsize=10)

    plt.tight_layout()
    
    # Save the figure
    save_path = out_dir / f"DL_Visualization_{axis.upper()}{slice_index}.png"
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization complete! Saved to: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot one vertical section of a generated model and its labels.")
    parser.add_argument("model_dir", nargs="?", default="outputs/dataset/model_0000")
    parser.add_argument("--axis", choices=["x", "y"], default="y", help="axis held constant (default y)")
    parser.add_argument("--index", type=int, default=128, help="index along --axis (default 128)")
    args = parser.parse_args()
    visualize_dl_model(args.model_dir, slice_index=args.index, axis=args.axis)
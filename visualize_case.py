import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

def load_case(case_path):
    """Load all modalities and segmentation for a single case."""
    case_id = os.path.basename(case_path)
    t1_path = os.path.join(case_path, f"{case_id}_t1.nii")
    t1ce_path = os.path.join(case_path, f"{case_id}_t1ce.nii")
    t2_path = os.path.join(case_path, f"{case_id}_t2.nii")
    flair_path = os.path.join(case_path, f"{case_id}_flair.nii")
    seg_path = os.path.join(case_path, f"{case_id}_seg.nii")
    
    t1 = nib.load(t1_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    return t1, t1ce, t2, flair, seg

def normalize_slice(slice_data):
    """Normalize slice data to [0, 1] range."""
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)
    return (slice_data - min_val) / (max_val - min_val)

def visualize_case(case_path, slice_indices=None):
    """Visualize a case with all modalities and segmentation for specified slices."""
    t1, t1ce, t2, flair, seg = load_case(case_path)
    
    if slice_indices is None:
        # Choose 5 slices evenly distributed throughout the brain
        total_slices = t1.shape[2]
        slice_indices = [
            total_slices // 5,      # Anterior
            2 * total_slices // 5,  # Anterior-middle
            total_slices // 2,      # Middle
            3 * total_slices // 5,  # Middle-posterior
            4 * total_slices // 5   # Posterior
        ]
    
    # Create figure with extra space for legend
    fig = plt.figure(figsize=(20, 5*len(slice_indices) + 3))
    gs = fig.add_gridspec(len(slice_indices) + 1, 5, height_ratios=[1]*len(slice_indices) + [0.3])
    
    # Create subplots for images
    axes = []
    for i in range(len(slice_indices)):
        row_axes = []
        for j in range(5):  # 4 modalities + segmentation
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Create legend subplot
    legend_ax = fig.add_subplot(gs[-1, :])
    legend_ax.axis('off')
    
    # Define segmentation classes and their colors
    seg_classes = {
        0: ('Background', 'black'),
        1: ('Necrotic and non-enhancing tumor core (NCR/NET)', 'blue'),
        2: ('Peritumoral edema (ED)', 'green'),
        3: ('GD-enhancing tumor (ET)', 'red')
    }
    
    # Create custom colormap for segmentation
    colors = ['black', 'blue', 'green', 'red']
    cmap = ListedColormap(colors)
    
    # Create legend patches with larger size
    legend_elements = [Patch(facecolor=color, label=label, edgecolor='black', linewidth=1) 
                      for label, color in seg_classes.values()]
    
    # Add legend with custom formatting
    legend = legend_ax.legend(handles=legend_elements, 
                            loc='center', 
                            ncol=2,
                            bbox_to_anchor=(0.5, 0.5),
                            fontsize=12,
                            frameon=True,
                            framealpha=0.9,
                            edgecolor='black')
    
    # Add title for legend
    legend_ax.set_title('Segmentation Classes', pad=20, fontsize=14, fontweight='bold')
    
    for idx, slice_idx in enumerate(slice_indices):
        # T1
        t1_slice = normalize_slice(t1[:, :, slice_idx])
        axes[idx][0].imshow(t1_slice, cmap='gray')
        axes[idx][0].set_title(f'T1 - Slice {slice_idx}', fontsize=12, pad=10)
        axes[idx][0].axis('off')
        
        # T1ce
        t1ce_slice = normalize_slice(t1ce[:, :, slice_idx])
        axes[idx][1].imshow(t1ce_slice, cmap='gray')
        axes[idx][1].set_title(f'T1ce - Slice {slice_idx}', fontsize=12, pad=10)
        axes[idx][1].axis('off')
        
        # T2
        t2_slice = normalize_slice(t2[:, :, slice_idx])
        axes[idx][2].imshow(t2_slice, cmap='gray')
        axes[idx][2].set_title(f'T2 - Slice {slice_idx}', fontsize=12, pad=10)
        axes[idx][2].axis('off')
        
        # FLAIR
        flair_slice = normalize_slice(flair[:, :, slice_idx])
        axes[idx][3].imshow(flair_slice, cmap='gray')
        axes[idx][3].set_title(f'FLAIR - Slice {slice_idx}', fontsize=12, pad=10)
        axes[idx][3].axis('off')
        
        # Segmentation with custom colormap
        seg_slice = seg[:, :, slice_idx]
        axes[idx][4].imshow(seg_slice, cmap=cmap, vmin=0, vmax=3)
        axes[idx][4].set_title(f'Segmentation - Slice {slice_idx}', fontsize=12, pad=10)
        axes[idx][4].axis('off')
    
    plt.tight_layout()
    plt.savefig('case_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage with a case from the training data
    case_path = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_366"
    visualize_case(case_path) 
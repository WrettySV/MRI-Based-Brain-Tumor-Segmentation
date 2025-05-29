import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
from torch.utils.data import DataLoader
from data_generator import BrainTumorDataset
from model import UNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
CHECKPOINT_PATH = 'model_checkpoints_unet_4/checkpoint_epoch_19.pt'
NUM_EXAMPLES = 8  # Number of examples to show
USE_VALIDATION = True  # Set to True to use validation data, False for training data

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

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    model = UNet(n_channels=4, n_classes=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("\nLoading model weights...")
    print(f"Checkpoint contains keys: {checkpoint.keys()}")
    print(f"Model state dict keys: {checkpoint['model_state_dict'].keys()}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")
    model.eval()
    return model

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice coefficient for each class
    """
    y_pred = F.softmax(y_pred, dim=1)
    y_true = y_true.to(y_pred.device)
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    intersection = (y_pred * y_true).sum(dim=2)
    union = y_pred.sum(dim=2) + y_true.sum(dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)
    # dice: shape [batch, n_classes]
    return dice[0]  # Dice for first sample in batch

def find_interesting_slices(dataset, num_slices=3):
    """Find slices with most classes or closest to center"""
    print("Analyzing slices to find most interesting ones...")
    
    # Get all slices for the case
    all_slices = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        # Count number of classes in this slice (excluding background)
        num_classes = torch.sum(torch.argmax(y, dim=0) > 0).item()
        # Calculate distance from center (assuming 16 slices total)
        center_distance = abs(i - 8)  # 8 is the middle slice (0-15)
        all_slices.append({
            'slice_idx': i,
            'num_classes': num_classes,
            'center_distance': center_distance,
            'has_tumor': num_classes > 0
        })
    
    # Sort slices by number of classes (descending) and center distance (ascending)
    interesting_slices = sorted(
        all_slices,
        key=lambda x: (-x['num_classes'], x['center_distance'])
    )
    
    # Take top slices that have tumor
    selected_slices = [s['slice_idx'] for s in interesting_slices if s['has_tumor']][:num_slices]
    
    print(f"Selected slices: {selected_slices}")
    for idx in selected_slices:
        print(f"Slice {idx}: {all_slices[idx]['num_classes']} classes, distance from center: {all_slices[idx]['center_distance']}")
    
    return selected_slices

def main():
    # Load model
    print("Loading model from checkpoint...")
    model = load_model(CHECKPOINT_PATH)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = BrainTumorDataset(
        list_IDs=['BraTS20_Training_356'],  # You can change this to any case ID
        dim=(128, 128),
        n_channels=4,
        slices_per_volume=16,
        is_validation=USE_VALIDATION  # Use validation data if specified
    )
    
    # Find interesting slices
    selected_slices = find_interesting_slices(dataset, num_slices=NUM_EXAMPLES)
    
    with torch.no_grad():
        for slice_idx in selected_slices:
            # Get specific slice
            X, y = dataset[slice_idx]
            X = X.unsqueeze(0).to(device)  # Add batch dimension
            y = y.unsqueeze(0).to(device)  # Add batch dimension
            
            outputs = model(X)
            
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Get case ID
            case_id = dataset.list_IDs[0]
            
            # Print debug information
            print(f"\nCase ID: {case_id}, Slice: {slice_idx}")
            print(f"Input shape: {X.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Probabilities shape: {probs.shape}")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Unique values in predictions: {torch.unique(predictions)}")
            
            # Calculate max probability per class
            max_probs = torch.max(probs, dim=2)[0]  # max по высоте
            max_probs = torch.max(max_probs, dim=2)[0]  # max по ширине
            print(f"Max probability per class: {max_probs[0]}")
            
            # Calculate metrics
            dice_per_class = dice_coefficient(outputs, y).cpu().numpy().flatten()
            mean_dice = dice_per_class.mean()
            if len(dice_per_class) >= 4:
                print(f"Dice per class: NCR/NET={dice_per_class[1]:.4f}, ED={dice_per_class[2]:.4f}, ET={dice_per_class[3]:.4f}")
            else:
                print(f"Dice per class: {dice_per_class}")
            print(f"Mean Dice: {mean_dice:.4f}")
            
            # Show predictions and ground truth
            plt.figure(figsize=(25, 10))
            plt.suptitle(f'Case: {case_id} - Slice {slice_idx} - Segmentation Results', fontsize=16)
            
            # First row: Ground Truth
            # Original FLAIR
            plt.subplot(251)
            plt.imshow(X[0, 3].cpu(), cmap='gray')
            plt.title('Original FLAIR')
            plt.colorbar()
            
            # Ground truth segmentation
            plt.subplot(252)
            plt.imshow(torch.argmax(y[0], dim=0).cpu(), cmap=cmap, vmin=0, vmax=3)
            plt.title('Ground Truth (All Classes)')
            plt.colorbar()
            
            # Show individual classes in ground truth
            for i in range(3):  # Show only tumor classes (1, 2, 3)
                plt.subplot(2, 5, i+3)
                mask = (torch.argmax(y[0], dim=0) == i+1).cpu()
                plt.imshow(mask, cmap='hot')
                plt.title(f'Ground Truth Class {i+1}')
            
            # Second row: Predictions
            # Original FLAIR (same as above)
            plt.subplot(256)
            plt.imshow(X[0, 3].cpu(), cmap='gray')
            plt.title('Original FLAIR')
            plt.colorbar()
            
            # Predicted segmentation
            plt.subplot(257)
            plt.imshow(predictions[0].cpu(), cmap=cmap, vmin=0, vmax=3)
            plt.title('Prediction (All Classes)')
            plt.colorbar()
            
            # Show individual classes in prediction
            for i in range(3):  # Show only tumor classes (1, 2, 3)
                plt.subplot(2, 5, i+8)
                mask = (predictions[0] == i+1).cpu()
                plt.imshow(mask, cmap='hot')
                plt.title(f'Predicted Class {i+1}')
            
            # Create a colored legend with Dice metrics in the labels
            if len(dice_per_class) >= 4:
                legend_elements = [
                    Patch(facecolor='black', label=f'Background'),
                    Patch(facecolor='blue', label=f'NCR/NET  Dice: {dice_per_class[1]:.4f}'),
                    Patch(facecolor='green', label=f'ED       Dice: {dice_per_class[2]:.4f}'),
                    Patch(facecolor='red', label=f'ET       Dice: {dice_per_class[3]:.4f}')
                ]
                mean_label = f"Mean Dice: {mean_dice:.4f}"
            else:
                legend_elements = []
                mean_label = f"Dice per class: {dice_per_class}\nMean: {mean_dice:.4f}"
            leg = plt.figlegend(
                handles=legend_elements,
                loc='lower left',
                bbox_to_anchor=(0.02, 0.02),  # very left, very bottom
                fontsize=12,
                frameon=True,
                title=mean_label
            )
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1.5)

            plt.tight_layout()
            plt.savefig(f'visualization_segmentation_{case_id}_slice_{slice_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("-" * 50)  # Separator between cases

if __name__ == '__main__':
    main() 
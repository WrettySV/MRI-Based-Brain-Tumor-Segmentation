import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from model import UNet, dice_coefficient, categorical_dice_loss
from data_generator import BrainTumorDataset
import gc


# Constants
TRAIN_PATH = '../../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
BATCH_SIZE = 2  # Keep small batch size
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.2  # 20% for validation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_case_ids():
    """Get list of all case IDs from the training directory"""
    case_ids = []
    for case_id in os.listdir(TRAIN_PATH):
        if case_id.startswith('BraTS20_Training_'):
            case_path = os.path.join(TRAIN_PATH, case_id)
            if all(os.path.exists(os.path.join(case_path, f'{case_id}_{mod}.nii')) 
                  for mod in ['t1', 't1ce', 't2', 'flair', 'seg']):  # Added t1 and t2
                case_ids.append(case_id)
    return sorted(case_ids)

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice coefficient for tumor classes
    
    Args:
        y_pred: Model output [batch_size, n_classes, height, width]
        y_true: Ground truth [batch_size, n_classes, height, width] (one-hot encoded)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice: Dice coefficients for each tumor class [NCR/NET, ED, ET]
        mean_dice: Mean Dice across tumor classes
    """
    # Convert predictions to probabilities
    y_pred = F.softmax(y_pred, dim=1)
    
    # Calculate intersection and union for each tumor class
    intersection = torch.sum(y_pred[:, 1:] * y_true[:, 1:], dim=(0, 2, 3))  # Skip background
    union = torch.sum(y_pred[:, 1:], dim=(0, 2, 3)) + torch.sum(y_true[:, 1:], dim=(0, 2, 3))
    
    # Calculate Dice coefficient for each tumor class
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice, dice.mean()

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    class_dices = torch.zeros(3).to(device)  # [NCR/NET, ED, ET]
    
    # Create progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X)
        
        # Calculate loss
        loss = categorical_dice_loss(outputs, y)
        dice_per_class, dice_mean = dice_coefficient(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_dice += dice_mean.item()
        class_dices += dice_per_class
        
        # Update progress bar with per-class Dice scores
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_mean.item():.4f}',
            'NCR/NET': f'{dice_per_class[0].item():.4f}',
            'ED': f'{dice_per_class[1].item():.4f}',
            'ET': f'{dice_per_class[2].item():.4f}'
        })
    
    # Calculate average per-class Dice scores
    class_dices /= len(train_loader)
    
    return total_loss / len(train_loader), total_dice / len(train_loader), class_dices

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    class_dices = torch.zeros(3).to(device)  # [NCR/NET, ED, ET]
    
    # Create progress bar
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            outputs = model(X)
            
            # Calculate metrics
            loss = categorical_dice_loss(outputs, y)
            dice_per_class, dice_mean = dice_coefficient(outputs, y)
            
            # Update metrics
            total_loss += loss.item()
            total_dice += dice_mean.item()
            class_dices += dice_per_class
            
            # Update progress bar with per-class Dice scores
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_mean.item():.4f}',
                'NCR/NET': f'{dice_per_class[0].item():.4f}',
                'ED': f'{dice_per_class[1].item():.4f}',
                'ET': f'{dice_per_class[2].item():.4f}'
            })
    
    # Calculate average per-class Dice scores
    class_dices /= len(val_loader)
    
    return total_loss / len(val_loader), total_dice / len(val_loader), class_dices

class TrainingPlotter:
    def __init__(self):
        # Create a single figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 15))
        self.fig.suptitle('Training Progress', fontsize=16)
        
        # Create directory for plots if it doesn't exist
        self.plots_dir = 'training_plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize data storage
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_class_dices = {
            'NCR/NET': [],
            'ED': [],
            'ET': []
        }
        self.val_class_dices = {
            'NCR/NET': [],
            'ED': [],
            'ET': []
        }
        
        # Set up plots
        self.setup_plots()
        
    def setup_plots(self):
        # Loss plot
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        # Overall Dice plot
        self.ax2.set_title('Overall Dice Coefficient')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Dice')
        self.ax2.grid(True)
        
        # Class-specific Dice plot
        self.ax3.set_title('Class-specific Dice Coefficients')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Dice')
        self.ax3.grid(True)
        
        # Set y-axis limits
        self.ax1.set_ylim(0, 2)  # Loss can be higher than 1
        self.ax2.set_ylim(0, 1)
        self.ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
    def update(self, epoch, train_loss, val_loss, train_dice, val_dice, 
               train_class_dices, val_class_dices):
        # Update data
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_dices.append(train_dice)
        self.val_dices.append(val_dice)
        
        # Update class-specific Dice scores
        for i, (name, _) in enumerate(self.train_class_dices.items()):
            self.train_class_dices[name].append(train_class_dices[i].item())
            self.val_class_dices[name].append(val_class_dices[i].item())
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Losses
        self.ax1.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        self.ax1.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Plot 2: Overall Dice
        self.ax2.plot(self.train_dices, label='Train Dice', color='blue', linewidth=2)
        self.ax2.plot(self.val_dices, label='Val Dice', color='red', linewidth=2)
        self.ax2.set_title('Overall Dice Coefficient')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Dice')
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Plot 3: Class-specific Dice
        colors = ['blue', 'orange', 'green'] # NCR/NET (blue), ED (yellow), ET (red)
        for i, (name, _) in enumerate(self.train_class_dices.items()):
            self.ax3.plot(self.train_class_dices[name], 
                         label=f'Train {name}', color=colors[i], linewidth=2)
            self.ax3.plot(self.val_class_dices[name], 
                         label=f'Val {name}', color=colors[i], linestyle='--', linewidth=2)
        
        self.ax3.set_title('Class-specific Dice Coefficients')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Dice')
        self.ax3.grid(True)
        self.ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add current values as text
        self.ax1.text(0.02, 0.95, f'Current Train Loss: {train_loss:.4f}\nCurrent Val Loss: {val_loss:.4f}',
                     transform=self.ax1.transAxes, verticalalignment='top')
        self.ax2.text(0.02, 0.95, f'Current Train Dice: {train_dice:.4f}\nCurrent Val Dice: {val_dice:.4f}',
                     transform=self.ax2.transAxes, verticalalignment='top')
        
        # Add epoch number to the main title
        self.fig.suptitle(f'Training Progress - Epoch {epoch+1}', fontsize=16)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Save current plot
        self.save_current_plot(epoch)
        
    def save_current_plot(self, epoch):
        """Save the current plot with epoch number"""
        filename = os.path.join(self.plots_dir, f'training_progress_epoch_{epoch+1:03d}.png')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    def save(self, filename='training_progress_final.png'):
        """Save the final plot"""
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Get all case IDs
    case_ids = get_case_ids()
    print(f"Found {len(case_ids)} valid cases")
    
    # Split into train and validation
    val_size = int(len(case_ids) * VAL_SPLIT)
    train_size = len(case_ids) - val_size
    train_ids, val_ids = random_split(case_ids, [train_size, val_size])
    
    print(f"Train set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create datasets with 4 channels and original dimensions
    train_dataset = BrainTumorDataset(train_ids, dim=(128, 128), n_channels=4, slices_per_volume=16)  
    val_dataset = BrainTumorDataset(val_ids, dim=(128, 128), n_channels=4, slices_per_volume=16)
    
    # Create data loaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Print batch dimensions
    for X, y in train_loader:
        print("\nFirst batch dimensions:")
        print(f"Input (X) shape: {X.shape}")  # [batch_size, 4, height, width]
        print(f"Target (y) shape: {y.shape}")  # [batch_size, classes, height, width]
        break
    
    # Initialize model with 4 input channels
    model = UNet(n_channels=4, n_classes=4).to(DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add warmup scheduler
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # Main scheduler with gentler learning rate reduction
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.8,
        patience=7,  
        min_lr=1e-6
    )
    
    # Initialize plotter
    plotter = TrainingPlotter()
    
    # Create directory for model checkpoints
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        try:
            # Train
            train_loss, train_dice, train_class_dices = train_epoch(model, train_loader, optimizer, DEVICE)
            
            # Validate
            val_loss, val_dice, val_class_dices = validate(model, val_loader, DEVICE)
            
            # Update plots
            plotter.update(epoch, train_loss, val_loss, train_dice, val_dice,
                          train_class_dices, val_class_dices)
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
            print("Train per-class Dice:")
            print(f"  NCR/NET: {train_class_dices[0]:.4f}")
            print(f"  ED: {train_class_dices[1]:.4f}")
            print(f"  ET: {train_class_dices[2]:.4f}")
            
            print(f"\nVal   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            print("Val per-class Dice:")
            print(f"  NCR/NET: {val_class_dices[0]:.4f}")
            print(f"  ED: {val_class_dices[1]:.4f}")
            print(f"  ET: {val_class_dices[2]:.4f}")
            
            # Learning rate scheduling
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step(val_loss)
            
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nCurrent learning rate: {current_lr:.2e}")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
                'train_class_dices': train_class_dices,
                'val_class_dices': val_class_dices
            }
            torch.save(checkpoint, f'model_checkpoints/checkpoint_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved: model_checkpoints/checkpoint_epoch_{epoch+1}.pt")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM error in epoch {epoch+1}. Saving checkpoint and clearing memory...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                # Save the last checkpoint if we have one
                if 'checkpoint' in locals():
                    torch.save(checkpoint, f'model_checkpoints/checkpoint_epoch_{epoch+1}_OOM.pt')
                print("Continuing to next epoch...")
                continue
            else:
                raise e
        
        # Clear GPU memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final plot
    plotter.save()

if __name__ == "__main__":
    main() 
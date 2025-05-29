import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generator_sam import BrainTumorDatasetSAM, get_train_val_split
from sam_model import SAMBrainTumorModel
import os
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_dir,
    writer
):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save model checkpoints
        writer: TensorBoard writer
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate Dice score
            pred = torch.argmax(output, dim=1)
            dice = calculate_dice_score(pred, torch.argmax(target, dim=1))
            train_dice += dice
            
            # Log batch metrics
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Dice/train_batch', dice, epoch * len(train_loader) + batch_idx)
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # Calculate Dice score
                pred = torch.argmax(output, dim=1)
                dice = calculate_dice_score(pred, torch.argmax(target, dim=1))
                val_dice += dice
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Log epoch metrics
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Dice/train_epoch', train_dice, epoch)
        writer.add_scalar('Dice/val_epoch', val_dice, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

def calculate_dice_score(pred, target):
    """
    Calculate Dice score for each class and return mean
    
    Args:
        pred: Predicted segmentation (batch_size, height, width)
        target: Ground truth segmentation (batch_size, height, width)
    
    Returns:
        mean_dice: Mean Dice score across all classes
    """
    dice_scores = []
    
    for class_idx in range(4):  # 4 classes
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum() + target_mask.sum()
        
        if union > 0:
            dice = (2. * intersection) / union
            dice_scores.append(dice.item())
        else:
            dice_scores.append(0.0)
    
    return np.mean(dice_scores)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/brain_tumor_segmentation')
    
    # Get train/val split
    train_ids, val_ids = get_train_val_split(validation_ratio=0.2, seed=42)
    
    # Create datasets
    train_dataset = BrainTumorDatasetSAM(list_IDs=train_ids, is_validation=False)
    val_dataset = BrainTumorDatasetSAM(list_IDs=val_ids, is_validation=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SAMBrainTumorModel(
        model_type="vit_h",
        checkpoint_path="models/sam_vit_h.pth",  # Updated path to downloaded model
        device=device
    )
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        device=device,
        save_dir=save_dir,
        writer=writer
    )
    
    writer.close()

if __name__ == '__main__':
    main()
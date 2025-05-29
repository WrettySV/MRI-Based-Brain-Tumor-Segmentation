import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.data import Dataset, CacheDataset
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import numpy as np
from pathlib import Path
from swin_unetr_model import get_swin_unetr_model

def get_transforms(mode="train"):
    """Get transforms for training or validation"""
    if mode == "train":
        return Compose([
            LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            Orientationd(keys=["t1", "t1ce", "t2", "flair", "seg"], axcodes="RAS"),
            Spacingd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["t1", "t1ce", "t2", "flair"],
                a_min=0.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                source_key="t1",
            ),
            RandCropByPosNegLabeld(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                label_key="seg",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandAffined(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                prob=0.15,
                rotate_range=(0.05, 0.05, 0.05),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandFlipd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                spatial_axis=[0],
                prob=0.5,
            ),
            RandFlipd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                spatial_axis=[1],
                prob=0.5,
            ),
            RandFlipd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                spatial_axis=[2],
                prob=0.5,
            ),
            RandRotate90d(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                prob=0.5,
                max_k=3,
            ),
            ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        ])
    else:
        return Compose([
            LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            Orientationd(keys=["t1", "t1ce", "t2", "flair", "seg"], axcodes="RAS"),
            Spacingd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["t1", "t1ce", "t2", "flair"],
                a_min=0.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                source_key="t1",
            ),
            ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        ])

def get_data_dicts(data_dir, val_split=0.2):
    """Get data dictionaries for MONAI dataset and split into train/val"""
    data_dicts = []
    for case_dir in Path(data_dir).iterdir():
        if case_dir.is_dir():
            case_id = case_dir.name
            # Check if all required files exist
            required_files = [
                f"{case_id}_t1.nii",
                f"{case_id}_t1ce.nii",
                f"{case_id}_t2.nii",
                f"{case_id}_flair.nii",
                f"{case_id}_seg.nii"
            ]
            if all((case_dir / f).exists() for f in required_files):
                data_dict = {
                    "t1": str(case_dir / f"{case_id}_t1.nii"),
                    "t1ce": str(case_dir / f"{case_id}_t1ce.nii"),
                    "t2": str(case_dir / f"{case_id}_t2.nii"),
                    "flair": str(case_dir / f"{case_id}_flair.nii"),
                    "seg": str(case_dir / f"{case_id}_seg.nii"),
                }
                data_dicts.append(data_dict)
    
    # Shuffle data
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(data_dicts)
    
    # Split into train and validation
    val_size = int(len(data_dicts) * val_split)
    train_dicts = data_dicts[val_size:]
    val_dicts = data_dicts[:val_size]
    
    return train_dicts, val_dicts

def train_unetr(
    data_dir,
    output_dir,
    num_epochs=100,
    batch_size=1,
    learning_rate=1e-4,
    device="cuda",
    num_workers=4,
    cache_rate=1.0,
    val_split=0.2,
):
    """Train Swin UNETR model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data dictionaries
    train_dicts, val_dicts = get_data_dicts(data_dir, val_split=val_split)
    
    print(f"Found {len(train_dicts)} training cases and {len(val_dicts)} validation cases")
    
    # Create datasets
    train_transforms = get_transforms(mode="train")
    val_transforms = get_transforms(mode="val")
    
    # Create datasets with proper transforms
    train_ds = Dataset(
        data=train_dicts,
        transform=train_transforms,
    )
    
    val_ds = Dataset(
        data=val_dicts,
        transform=val_transforms,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Debug print to check data structure
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Batch shapes:")
        for k, v in batch.items():
            print(f"{k}: {v.shape}")
        break
    
    # Create model
    model = get_swin_unetr_model(
        img_size=(96, 96, 96),  # Adjusted for memory constraints
        in_channels=4,
        out_channels=4,
        feature_size=12,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,  # Enable gradient checkpointing to save memory
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function and optimizer
    loss_function = DiceCELoss(sigmoid=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    
    # Training loop
    best_val_dice = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Get input data
            t1 = batch["t1"].to(device)
            t1ce = batch["t1ce"].to(device)
            t2 = batch["t2"].to(device)
            flair = batch["flair"].to(device)
            seg = batch["seg"].to(device)
            
            # Combine modalities
            x = torch.cat([t1, t1ce, t2, flair], dim=1)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            
            # Calculate loss
            loss = loss_function(output, seg)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Get input data
                t1 = batch["t1"].to(device)
                t1ce = batch["t1ce"].to(device)
                t2 = batch["t2"].to(device)
                flair = batch["flair"].to(device)
                seg = batch["seg"].to(device)
                
                # Combine modalities
                x = torch.cat([t1, t1ce, t2, flair], dim=1)
                
                # Forward pass with sliding window
                output = sliding_window_inference(
                    x,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                )
                
                # Calculate metrics
                dice_metric(y_pred=output, y=seg)
        
        # Get validation metrics
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                },
                os.path.join(output_dir, "best_model.pth"),
            )
            print(f"New best model saved! (Val Dice: {val_dice:.4f})")
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Best Val Dice: {best_val_dice:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)

if __name__ == "__main__":
    train_unetr(
        data_dir="BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
        output_dir="output_unetr",
        num_epochs=100,
        batch_size=1,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=4,
        cache_rate=1.0,
        val_split=0.2,  # 20% of data for validation
    ) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, bilinear=False):
        """
        UNet model for brain tumor segmentation
        
        Args:
            n_channels: Number of input channels (FLAIR, T1, T1ce, T2)
            n_classes: Number of segmentation classes (background, necrotic core, edema, enhancing)
            bilinear: Whether to use bilinear interpolation for upsampling
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)  # 4 -> 64 channels
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024 // factor)
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512)  # 1024 = 512 (skip) + 512 (up)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256)  # 512 = 256 (skip) + 256 (up)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128)  # 256 = 128 (skip) + 128 (up)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)  # 128 = 64 (skip) + 64 (up)
        )
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, n_channels, height, width]
               n_channels = 4 (FLAIR, T1, T1ce, T2)
        
        Returns:
            Output tensor of shape [batch_size, n_classes, height, width]
            n_classes = 4 (background, necrotic core, edema, enhancing)
        """
        # Encoder
        x1 = self.inc(x)        # [B, 64, 128, 128]
        x2 = self.down1(x1)     # [B, 128, 64, 64]
        x3 = self.down2(x2)     # [B, 256, 32, 32]
        x4 = self.down3(x3)     # [B, 512, 16, 16]
        x5 = self.down4(x4)     # [B, 1024, 8, 8]

        # Decoder with skip connections
        x = self.up1[0](x5)     # [B, 512, 16, 16]
        x = torch.cat([x4, x], dim=1)  # [B, 1024, 16, 16]
        x = self.up1[1](x)      # [B, 512, 16, 16]
        
        x = self.up2[0](x)      # [B, 256, 32, 32]
        x = torch.cat([x3, x], dim=1)  # [B, 512, 32, 32]
        x = self.up2[1](x)      # [B, 256, 32, 32]
        
        x = self.up3[0](x)      # [B, 128, 64, 64]
        x = torch.cat([x2, x], dim=1)  # [B, 256, 64, 64]
        x = self.up3[1](x)      # [B, 128, 64, 64]
        
        x = self.up4[0](x)      # [B, 64, 128, 128]
        x = torch.cat([x1, x], dim=1)  # [B, 128, 128, 128]
        x = self.up4[1](x)      # [B, 64, 128, 128]
        
        logits = self.outc(x)   # [B, n_classes, 128, 128]
        return logits

# Loss functions
def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        y_pred: Model output [batch_size, n_classes, height, width]
        y_true: Ground truth [batch_size, n_classes, height, width] (one-hot encoded)
        smooth: Smoothing factor
    """
    # Convert predictions to probabilities
    y_pred = F.softmax(y_pred, dim=1)
    
    # Ensure tensors are on the same device
    y_true = y_true.to(y_pred.device)
    
    # Reshape tensors to [batch_size, n_classes, -1]
    y_pred = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
    y_true = y_true.view(y_true.size(0), y_true.size(1), -1)
    
    # Calculate intersection and union
    intersection = (y_pred * y_true).sum(dim=2)
    union = y_pred.sum(dim=2) + y_true.sum(dim=2)
    
    # Calculate Dice coefficient for each class
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return mean Dice coefficient across all classes
    return dice.mean()

# def dice_loss(y_pred, y_true):
#     # Clamp labels to valid range [0, 3]
#     y_true = torch.clamp(y_true, 0, 3)
#     return 1 - dice_coefficient(y_pred, y_true)

# def categorical_dice_loss(y_pred, y_true):
#     # y_pred: [batch_size, n_classes, height, width]
#     # y_true: [batch_size, height, width]
    
#     # Clamp labels to valid range [0, 3]
#     y_true = torch.clamp(y_true, 0, 3)
    
#     ce_loss = F.cross_entropy(y_pred, y_true)
#     dice = dice_loss(y_pred, y_true)
#     return ce_loss + dice

def dice_loss(y_pred, y_true):
    """
    Calculate Dice loss
    
    Args:
        y_pred: Model output [batch_size, n_classes, height, width]
        y_true: Ground truth [batch_size, n_classes, height, width] (one-hot encoded)
    """
    return 1 - dice_coefficient(y_pred, y_true)

def categorical_dice_loss(y_pred, y_true):
    """
    Combined categorical cross-entropy and Dice loss
    
    Args:
        y_pred: Model output [batch_size, n_classes, height, width]
        y_true: Ground truth [batch_size, n_classes, height, width] (one-hot encoded)
    """
    # Convert one-hot to class indices for cross entropy
    y_class = torch.argmax(y_true, dim=1)
    
    # Calculate losses
    ce_loss = F.cross_entropy(y_pred, y_class)
    dice = dice_loss(y_pred, y_true)
    return ce_loss + dice

def visualize_prediction(X, y_pred, y_true, case_id=None):
    """
    Visualize model prediction
    
    Args:
        X: Input tensor [batch_size, n_channels, height, width]
        y_pred: Model output [batch_size, n_classes, height, width]
        y_true: Ground truth [batch_size, height, width]
        case_id: Optional case ID to display in the title
    """
    # Get probabilities
    probs = F.softmax(y_pred, dim=1)
    
    # Get final segmentation
    pred_seg = torch.argmax(probs, dim=1)
    true_seg = torch.argmax(y_true, dim=1)  # Convert one-hot to class indices
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Add case ID to the main title if provided
    if case_id:
        fig.suptitle(f'Case: {case_id}', fontsize=16)
    
    # Input images
    axes[0, 0].imshow(X[0, 0].cpu(), cmap='gray')
    axes[0, 0].set_title('FLAIR')
    
    axes[0, 1].imshow(X[0, 1].cpu(), cmap='gray')
    axes[0, 1].set_title('T1')
    
    axes[0, 2].imshow(X[0, 2].cpu(), cmap='gray')
    axes[0, 2].set_title('T1ce')
    
    axes[0, 3].imshow(X[0, 3].cpu(), cmap='gray')
    axes[0, 3].set_title('T2')
    
    # Probabilities for each class
    for i in range(4):
        axes[0, i+4].imshow(probs[0, i].cpu(), cmap='hot')
        axes[0, i+4].set_title(f'Class {i} Probability')
    
    # Final segmentation
    axes[1, 0].imshow(pred_seg[0].cpu(), cmap='tab10')
    axes[1, 0].set_title('Predicted Segmentation')
    
    # Ground truth visualization
    axes[1, 1].imshow(true_seg[0].cpu(), cmap='tab10')
    axes[1, 1].set_title('Ground Truth Segmentation')
    
    # Show individual classes in ground truth
    for i in range(3):  # Show only tumor classes (1, 2, 3)
        mask = (true_seg[0] == i+1).cpu()
        axes[1, i+2].imshow(mask, cmap='hot')
        axes[1, i+2].set_title(f'Class {i+1} in Ground Truth')
    
    plt.tight_layout()
    plt.show()

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=50):
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Lists to store metrics
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        train_loss = 0
        train_dice = 0
        train_ce_loss = 0
        train_dice_loss = 0
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_idx, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            
            # Calculate losses
            loss = categorical_dice_loss(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, y).item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_coefficient(outputs, y).item():.4f}'
            })
            
            # Visualize first batch of first epoch
            if epoch == 0 and batch_idx == 0:
                # Get case ID from the dataset
                case_id = train_loader.dataset.valid_cases[0]
                visualize_prediction(X, outputs, y, case_id)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        
        # Validation loop with progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                
                # Calculate losses
                loss = categorical_dice_loss(outputs, y)
                
                # Update metrics
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, y).item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice_coefficient(outputs, y).item():.4f}'
                })
        
        # Calculate averages
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss:.4f}, Dice Coef: {train_dice:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Dice Coef: {val_dice:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return {
        'train_losses': train_losses,
        'train_dices': train_dices,
        'val_losses': val_losses,
        'val_dices': val_dices
    } 
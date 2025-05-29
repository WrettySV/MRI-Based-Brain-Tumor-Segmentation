import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.networks.layers import Norm
from monai.utils import ensure_tuple_rep

class SwinUNETRModel(nn.Module):
    def __init__(
        self,
        img_size=(240, 240, 155),
        in_channels=4,
        out_channels=4,
        feature_size=12,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    ):
        """
        Swin UNETR model for BraTS segmentation
        
        Args:
            img_size: Input image size (H, W, D)
            in_channels: Number of input channels (4 for BraTS: T1, T1ce, T2, FLAIR)
            out_channels: Number of output channels (4 for BraTS: background, NCR/NET, ED, ET)
            feature_size: Feature size for Swin Transformer
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Dropout path rate
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
                where C=4 for BraTS modalities
            
        Returns:
            Output tensor of shape (B, 4, H, W, D)
                where 4 represents the segmentation classes
        """
        return self.model(x)

def get_swin_unetr_model(
    img_size=(240, 240, 155),
    in_channels=4,
    out_channels=4,
    feature_size=12,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
):
    """
    Factory function to create Swin UNETR model
    
    Args:
        img_size: Input image size (H, W, D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        feature_size: Feature size for Swin Transformer
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        dropout_path_rate: Dropout path rate
        use_checkpoint: Whether to use gradient checkpointing
    
    Returns:
        SwinUNETRModel instance
    """
    model = SwinUNETRModel(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    )
    return model 
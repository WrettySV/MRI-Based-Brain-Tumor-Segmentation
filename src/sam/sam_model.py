import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

class SAMBrainTumorModel(nn.Module):
    def __init__(self, model_type="vit_h", checkpoint_path=None, device=None):
        """
        Initialize SAM model for brain tumor segmentation
        
        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', or 'vit_b')
            checkpoint_path: Path to SAM checkpoint
            device: Device to run model on
        """
        super().__init__()
        
        # Initialize SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        
        # Freeze SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False
            
        # Add segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1)  # 4 classes: background, necrotic, edema, enhancing
        )
        
        # Enable gradient checkpointing
        self.sam.image_encoder.gradient_checkpointing_enable()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 4, height, width)
               4 channels: T1, T1ce, T2, FLAIR
            
        Returns:
            output: Segmentation logits of shape (batch_size, num_classes, height, width)
        """
        # Convert 4 channels to RGB for SAM
        # Use T1ce for red channel (enhancing tumor)
        # Use T2 for green channel (edema)
        # Use FLAIR for blue channel (overall tumor)
        rgb = torch.stack([
            x[:, 1],  # T1ce -> Red
            x[:, 2],  # T2 -> Green
            x[:, 3],  # FLAIR -> Blue
        ], dim=1)
        
        # Get image embeddings from SAM
        image_embedding = self.sam.image_encoder(rgb)
        
        # Pass through segmentation head
        output = self.segmentation_head(image_embedding)
        
        return output 
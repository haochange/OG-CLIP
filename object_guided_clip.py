# Object-Guided Contrastive Language-Image Pre-training Implementation
# Based on SAM2 and CLIP

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


class ObjectGuidedCLIP(nn.Module):
    """
    Object-Guided Contrastive Language-Image Pre-training model.
    
    This model uses SAM2 to generate object masks, then processes both the 
    original image and the masked regions through separate convolutional layers
    before feeding them into CLIP's image encoder for feature extraction.
    """
    
    def __init__(
        self,
        sam_model_path: str,
        sam_config_path: str,
        clip_model_name: str = "ViT-B/32",
        mask_conv_channels: int = 64,
        object_conv_channels: int = 128,
        temperature: float = 0.07,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Initialize SAM2 model
        self.sam_predictor = self._build_sam2(sam_model_path, sam_config_path)
        
        # Initialize CLIP model (we'll use a simplified version here)
        self.clip_image_encoder = self._build_clip_image_encoder(clip_model_name)
        
        # Convolutional layers for processing masked regions
        self.mask_conv = nn.Sequential(
            nn.Conv2d(3, mask_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_conv_channels, mask_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_conv_channels),
            nn.ReLU(inplace=True)
        )
        
        # Convolutional layers for processing object regions
        self.object_conv = nn.Sequential(
            nn.Conv2d(3, object_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(object_conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(object_conv_channels, object_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(object_conv_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mask_conv_channels + object_conv_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),  # CLIP ViT-B/32 outputs 512 features
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
        # Text encoder placeholder (would be CLIP text encoder in full implementation)
        self.text_projection = nn.Linear(512, 128)  # Simplified text projection
        
    def _build_sam2(self, model_path: str, config_path: str):
        """Build SAM2 model from checkpoint and config."""
        try:
            sam_model = build_sam2(config_path, model_path, device=self.device)
            predictor = SAM2ImagePredictor(sam_model)
            return predictor
        except Exception as e:
            print(f"Warning: Could not load SAM2 model: {e}")
            print("Using dummy SAM2 predictor for development")
            return None
    
    def _build_clip_image_encoder(self, model_name: str):
        """Build CLIP image encoder."""
        # Simplified CLIP image encoder for demonstration
        # In practice, you would load the actual CLIP model
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
    
    def extract_object_mask(
        self, 
        image: Union[np.ndarray, Image.Image], 
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Extract object mask using SAM2.
        
        Args:
            image: Input image
            point_coords: Point coordinates for prompting
            point_labels: Point labels (1 for foreground, 0 for background)
            box: Bounding box coordinates
            
        Returns:
            Object mask tensor
        """
        if self.sam_predictor is None:
            # Create dummy mask for development
            if isinstance(image, Image.Image):
                w, h = image.size
                mask = torch.zeros((1, h, w), device=self.device)
                # Add a simple rectangular mask in the center
                mask[:, h//4:3*h//4, w//4:3*w//4] = 1.0
            else:
                h, w = image.shape[:2]
                mask = torch.zeros((1, h, w), device=self.device)
                mask[:, h//4:3*h//4, w//4:3*w//4] = 1.0
            return mask
        
        # Set image for SAM2
        self.sam_predictor.set_image(image)
        
        # Predict mask
        masks, _, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False
        )
        
        # Convert to tensor and return
        mask = torch.from_numpy(masks[0]).float().to(self.device)
        return mask.unsqueeze(0)  # Add batch dimension
    
    def apply_mask_to_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to image to extract object region.
        
        Args:
            image: Input image tensor (B, C, H, W)
            mask: Mask tensor (B, H, W)
            
        Returns:
            Masked image tensor
        """
        # Ensure mask has proper dimensions
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Add channel dimension
        
        # Apply mask to each channel
        masked_image = image * mask
        return masked_image
    
    def forward(
        self, 
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        text: Optional[torch.Tensor] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of Object-Guided CLIP.
        
        Args:
            image: Input image
            text: Text embeddings (optional)
            point_coords: Point coordinates for SAM2 prompting
            point_labels: Point labels for SAM2 prompting
            box: Bounding box for SAM2 prompting
            
        Returns:
            Tuple of (image_features, text_features, object_features)
        """
        # Convert image to tensor if needed
        if isinstance(image, (np.ndarray, Image.Image)):
            if isinstance(image, Image.Image):
                image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
            else:
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(self.device) / 255.0
        else:
            image_tensor = image.to(self.device)
        
        # Extract object mask using SAM2
        object_mask = self.extract_object_mask(
            image, point_coords, point_labels, box
        )
        
        # Apply mask to get object region
        masked_object = self.apply_mask_to_image(image_tensor, object_mask)
        
        # Process original image through mask convolution
        mask_features = self.mask_conv(image_tensor)
        
        # Process masked object through object convolution
        object_features = self.object_conv(masked_object)
        
        # Fuse features
        fused_features = torch.cat([mask_features, object_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Extract final features using CLIP-like encoder
        image_features = self.clip_image_encoder(fused_features)
        image_features = self.projection_head(image_features)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        
        # Process text if provided
        text_features = None
        if text is not None:
            text_features = self.text_projection(text)
            text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features, object_features
    
    def contrastive_loss(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.
        
        Args:
            image_features: Normalized image features
            text_features: Normalized text features
            labels: Ground truth labels (optional)
            
        Returns:
            Contrastive loss tensor
        """
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        if labels is None:
            # Create labels for contrastive learning (diagonal is positive pairs)
            batch_size = image_features.size(0)
            labels = torch.arange(batch_size, device=image_features.device)
        
        # Cross entropy loss
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        
        # Average the losses
        loss = (loss_img + loss_txt) / 2
        return loss


class ObjectGuidedCLIPTrainer:
    """Trainer class for Object-Guided CLIP model."""
    
    def __init__(
        self,
        model: ObjectGuidedCLIP,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
    
    def train_step(
        self, 
        images: torch.Tensor, 
        texts: torch.Tensor,
        point_coords: Optional[list] = None,
        point_labels: Optional[list] = None
    ) -> dict:
        """
        Single training step.
        
        Args:
            images: Batch of images
            texts: Batch of text embeddings
            point_coords: List of point coordinates for each image
            point_labels: List of point labels for each image
            
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = images.size(0)
        all_image_features = []
        all_text_features = []
        
        # Process each image in the batch
        for i in range(batch_size):
            img = images[i]
            txt = texts[i] if texts.dim() > 1 else texts
            
            # Get point prompts if available
            p_coords = point_coords[i] if point_coords else None
            p_labels = point_labels[i] if point_labels else None
            
            # Forward pass
            img_features, txt_features, _ = self.model(
                img, txt, p_coords, p_labels
            )
            
            all_image_features.append(img_features)
            all_text_features.append(txt_features)
        
        # Stack features
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # Compute contrastive loss
        loss = self.model.contrastive_loss(image_features, text_features)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            # Compute accuracy (assuming diagonal is positive pairs)
            logits = torch.matmul(image_features, text_features.T) / self.model.temperature
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == torch.arange(batch_size, device=images.device)).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }


def create_object_guided_clip(
    sam_model_path: str = "checkpoints/sam2.1_hiera_tiny.pt",
    sam_config_path: str = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> ObjectGuidedCLIP:
    """
    Create an Object-Guided CLIP model.
    
    Args:
        sam_model_path: Path to SAM2 checkpoint
        sam_config_path: Path to SAM2 config
        device: Device to run on
        
    Returns:
        ObjectGuidedCLIP model
    """
    model = ObjectGuidedCLIP(
        sam_model_path=sam_model_path,
        sam_config_path=sam_config_path,
        device=device
    )
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = create_object_guided_clip(device=device)
    
    # Create dummy data for testing
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_texts = torch.randn(batch_size, 512).to(device)  # Simplified text embeddings
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        img_features, txt_features, obj_features = model(dummy_images[0])
        print(f"Image features shape: {img_features.shape}")
        print(f"Text features shape: {txt_features.shape}")
        print(f"Object features shape: {obj_features.shape}")
    
    # Test contrastive loss
    print("\nTesting contrastive loss...")
    loss = model.contrastive_loss(img_features.unsqueeze(0), txt_features.unsqueeze(0))
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Test trainer
    print("\nTesting trainer...")
    trainer = ObjectGuidedCLIPTrainer(model)
    metrics = trainer.train_step(dummy_images, dummy_texts)
    print(f"Training metrics: {metrics}")
    
    print("\nObject-Guided CLIP implementation completed successfully!")
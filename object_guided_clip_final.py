# Object-Guided CLIP - Main Implementation
# Complete implementation of Object-Guided Contrastive Language-Image Pre-training

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2

# Import SAM2 components
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
except ImportError:
    print("Warning: SAM2 not available. Install SAM2 first.")
    SAM2ImagePredictor = None
    build_sam2 = None


class ObjectGuidedCLIP(nn.Module):
    """
    Object-Guided Contrastive Language-Image Pre-training model.
    
    This model combines SAM2 for object segmentation with CLIP for contrastive learning.
    Objects are segmented using SAM2, then both the original image and object masks
    are processed through convolutional layers before being fed into CLIP's image encoder.
    """
    
    def __init__(
        self,
        sam2_model: SAM2ImagePredictor,
        clip_image_encoder: nn.Module,
        clip_text_encoder: nn.Module,
        mask_conv_channels: int = 256,
        object_conv_channels: int = 256,
        feature_fusion_dim: int = 512,
        embedding_dim: int = 512,
        temperature: float = 0.07,
        image_size: int = 224
    ):
        """
        Initialize Object-Guided CLIP model.
        
        Args:
            sam2_model: SAM2 image predictor model
            clip_image_encoder: CLIP image encoder
            clip_text_encoder: CLIP text encoder
            mask_conv_channels: Number of channels for mask convolution
            object_conv_channels: Number of channels for object convolution
            feature_fusion_dim: Dimension for feature fusion
            embedding_dim: Final embedding dimension
            temperature: Temperature for contrastive learning
            image_size: Input image size
        """
        super().__init__()
        
        self.sam2_model = sam2_model
        self.clip_image_encoder = clip_image_encoder
        self.clip_text_encoder = clip_text_encoder
        
        self.temperature = temperature
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        
        # Convolutional layers for mask and object processing
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, mask_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_conv_channels, mask_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_conv_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))  # Match CLIP feature size
        )
        
        self.object_conv = nn.Sequential(
            nn.Conv2d(3, object_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(object_conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(object_conv_channels, object_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(object_conv_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((image_size // 32, image_size // 32))  # Match CLIP feature size
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(
                mask_conv_channels + object_conv_channels + 768,  # 768 is CLIP feature dim
                feature_fusion_dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(feature_fusion_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_fusion_dim, embedding_dim)
        )
        
        # Text projection head
        self.text_projection = nn.Sequential(
            nn.Linear(512, embedding_dim),  # CLIP text embedding size is 512
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Image projection head
        self.image_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        return_intermediate: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Forward pass of Object-Guided CLIP.
        
        Args:
            image: Input image [C, H, W]
            text: Text embedding [D]
            point_coords: Point coordinates for SAM2 [N, 2]
            point_labels: Point labels for SAM2 [N]
            return_intermediate: Whether to return intermediate features
        
        Returns:
            image_features: Image features [D]
            text_features: Text features [D]
            intermediate: Optional intermediate features
        """
        # Generate object mask using SAM2
        mask = self.generate_object_mask(image, point_coords, point_labels)
        
        # Process mask through convolutional layers
        mask_features = self.mask_conv(mask.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W] -> [1, C, H', W']
        
        # Process original image through object convolution
        image_resized = F.interpolate(
            image.unsqueeze(0), 
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        object_features = self.object_conv(image_resized)  # [1, C, H', W']
        
        # Get CLIP image features
        clip_image_features = self.get_clip_image_features(image_resized)
        
        # Concatenate all features
        fused_features = torch.cat([
            mask_features,
            object_features,
            clip_image_features
        ], dim=1)  # [1, C_total, H', W']
        
        # Apply feature fusion
        fused_embedding = self.feature_fusion(fused_features)  # [1, D]
        
        # Project image features
        image_features = self.image_projection(fused_embedding)  # [1, D]
        
        # Process text features
        text_features = self.text_projection(text.unsqueeze(0))  # [1, D]
        
        if return_intermediate:
            intermediate = {
                'mask': mask,
                'mask_features': mask_features,
                'object_features': object_features,
                'clip_image_features': clip_image_features,
                'fused_features': fused_features
            }
            return image_features.squeeze(0), text_features.squeeze(0), intermediate
        
        return image_features.squeeze(0), text_features.squeeze(0)
    
    def generate_object_mask(
        self,
        image: torch.Tensor,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> torch.Tensor:
        """
        Generate object mask using SAM2.
        
        Args:
            image: Input image [C, H, W]
            point_coords: Point coordinates [N, 2]
            point_labels: Point labels [N]
        
        Returns:
            mask: Object mask [H, W]
        """
        # Convert image to numpy format for SAM2
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Set image in SAM2 predictor
        self.sam2_model.set_image(image_np)
        
        # Predict mask
        masks, scores, logits = self.sam2_model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Select best mask based on score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Convert to tensor
        mask_tensor = torch.from_numpy(best_mask.astype(np.float32)).to(image.device)
        
        return mask_tensor
    
    def get_clip_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP image features.
        
        Args:
            image: Input image [1, C, H, W]
        
        Returns:
            features: CLIP features [1, C_clip, H', W']
        """
        with torch.no_grad():
            features = self.clip_image_encoder(image)
        
        return features
    
    def contrastive_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            image_features: Image features [B, D]
            text_features: Text features [B, D]
            labels: Optional labels [B]
        
        Returns:
            loss: Contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        if labels is None:
            # Create labels for contrastive learning (diagonal is positive pairs)
            labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute cross-entropy loss for both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Average the losses
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


class ObjectGuidedCLIPTrainer:
    """Trainer for Object-Guided CLIP model."""
    
    def __init__(
        self,
        model: ObjectGuidedCLIP,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        temperature: float = 0.07
    ):
        """
        Initialize trainer.
        
        Args:
            model: Object-Guided CLIP model
            learning_rate: Learning rate
            weight_decay: Weight decay
            temperature: Temperature for contrastive learning
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training metrics
        self.train_loss = []
        self.train_accuracy = []
    
    def train_step(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        point_coords: List[np.ndarray],
        point_labels: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            images: Batch of images [B, C, H, W]
            texts: Batch of text embeddings [B, D]
            point_coords: List of point coordinates
            point_labels: List of point labels
        
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        batch_size = images.size(0)
        total_loss = 0
        correct_predictions = 0
        
        # Process each sample in the batch
        all_image_features = []
        all_text_features = []
        
        for i in range(batch_size):
            img = images[i]
            txt = texts[i]
            
            # Forward pass
            img_features, txt_features = self.model(
                image=img,
                text=txt,
                point_coords=point_coords[i],
                point_labels=point_labels[i]
            )
            
            all_image_features.append(img_features.unsqueeze(0))
            all_text_features.append(txt_features.unsqueeze(0))
        
        # Stack features
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        # Compute contrastive loss
        loss = self.model.contrastive_loss(image_features, text_features)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy
        with torch.no_grad():
            # Normalize features
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            
            # Compute similarity matrix
            logits = torch.matmul(image_features, text_features.T) / self.temperature
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            targets = torch.arange(batch_size, device=logits.device)
            
            # Compute accuracy
            correct_predictions = (predictions == targets).sum().item()
            accuracy = correct_predictions / batch_size
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        return metrics


def create_object_guided_clip(
    sam2_config_path: str = "sam2_hiera_b+.yaml",
    sam2_checkpoint_path: str = "sam2_hiera_base_plus.pt",
    clip_model_name: str = "ViT-B/32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> ObjectGuidedCLIP:
    """
    Create Object-Guided CLIP model.
    
    Args:
        sam2_config_path: Path to SAM2 config file
        sam2_checkpoint_path: Path to SAM2 checkpoint
        clip_model_name: CLIP model name
        device: Device to load model on
    
    Returns:
        model: Object-Guided CLIP model
    """
    # Load SAM2 model
    if SAM2ImagePredictor is None:
        raise ImportError("SAM2 is not available. Please install SAM2 first.")
    
    sam2_model = build_sam2(sam2_config_path, sam2_checkpoint_path, device=device)
    
    # Create dummy CLIP encoders (in practice, load real CLIP model)
    # For demonstration, we create simple encoders
    class DummyCLIPEncoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(input_dim, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim)
            )
        
        def forward(self, x):
            return self.conv(x)
    
    clip_image_encoder = DummyCLIPEncoder(3, 768).to(device)
    clip_text_encoder = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512)
    ).to(device)
    
    # Create Object-Guided CLIP model
    model = ObjectGuidedCLIP(
        sam2_model=sam2_model,
        clip_image_encoder=clip_image_encoder,
        clip_text_encoder=clip_text_encoder,
        mask_conv_channels=256,
        object_conv_channels=256,
        feature_fusion_dim=512,
        embedding_dim=512,
        temperature=0.07,
        image_size=224
    ).to(device)
    
    return model


# Utility functions for data preprocessing
def preprocess_image(image: Union[str, np.ndarray, Image.Image], size: int = 224) -> torch.Tensor:
    """
    Preprocess image for Object-Guided CLIP.
    
    Args:
        image: Input image (path, numpy array, or PIL Image)
        size: Target size
    
    Returns:
        tensor: Preprocessed image tensor [C, H, W]
    """
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Normalize with ImageNet statistics
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor


def create_point_prompts_from_bbox(
    bbox: List[float],
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create point prompts from bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_size: Image size (height, width)
    
    Returns:
        point_coords: Point coordinates [N, 2]
        point_labels: Point labels [N]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Create center point
    point_coords = np.array([[center_x, center_y]])
    point_labels = np.array([1])  # Foreground
    
    return point_coords, point_labels


if __name__ == "__main__":
    # Test the implementation
    print("Testing Object-Guided CLIP implementation...")
    
    # Create dummy model (SAM2 not available in this environment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple test components
    class DummySAM2:
        def set_image(self, image):
            pass
        
        def predict(self, point_coords, point_labels, multimask_output=True):
            # Return dummy mask
            mask = np.ones((224, 224), dtype=np.float32)
            scores = np.array([0.9, 0.8, 0.7])
            logits = np.random.randn(3, 256, 256)
            return [mask, mask, mask], scores, logits
    
    class DummyCLIPImageEncoder(nn.Module):
        def forward(self, x):
            return torch.randn(1, 768, 7, 7).to(x.device)
    
    # Create model
    model = ObjectGuidedCLIP(
        sam2_model=DummySAM2(),
        clip_image_encoder=DummyCLIPImageEncoder(),
        clip_text_encoder=nn.Linear(512, 512),
        mask_conv_channels=256,
        object_conv_channels=256,
        feature_fusion_dim=512,
        embedding_dim=512,
        temperature=0.07,
        image_size=224
    ).to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    image = torch.randn(3, 224, 224).to(device)
    text = torch.randn(512).to(device)
    point_coords = np.array([[112, 112]])
    point_labels = np.array([1])
    
    img_features, txt_features = model(image, text, point_coords, point_labels)
    print(f"Image features shape: {img_features.shape}")
    print(f"Text features shape: {txt_features.shape}")
    
    # Test contrastive loss
    print("Testing contrastive loss...")
    batch_img = torch.randn(4, 512).to(device)
    batch_txt = torch.randn(4, 512).to(device)
    loss = model.contrastive_loss(batch_img, batch_txt)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    print("Implementation test completed successfully!")
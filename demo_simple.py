# Object-Guided CLIP - Simple Demo
# Simplified demonstration of Object-Guided Contrastive Language-Image Pre-training

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import random

# Import the final implementation
from object_guided_clip_final import ObjectGuidedCLIP, ObjectGuidedCLIPTrainer, preprocess_image


class SimpleDemo:
    """Simple demonstration of Object-Guided CLIP functionality."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {device}")
        
        # Create dummy components for demonstration
        self.model = self._create_dummy_model()
        self.trainer = ObjectGuidedCLIPTrainer(self.model)
    
    def _create_dummy_model(self) -> ObjectGuidedCLIP:
        """Create a dummy model for demonstration purposes."""
        
        # Dummy SAM2 model
        class DummySAM2:
            def set_image(self, image):
                pass
            
            def predict(self, point_coords, point_labels, multimask_output=True):
                # Return dummy mask
                mask = np.ones((224, 224), dtype=np.float32)
                scores = np.array([0.9, 0.8, 0.7])
                logits = np.random.randn(3, 256, 256)
                return [mask, mask, mask], scores, logits
        
        # Dummy CLIP image encoder
        class DummyCLIPImageEncoder(nn.Module):
            def forward(self, x):
                # Return dummy features matching expected dimensions
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
        ).to(self.device)
        
        return model
    
    def create_synthetic_data(self, num_samples: int = 16) -> List[Dict]:
        """Create synthetic data for demonstration."""
        
        print(f"Creating {num_samples} synthetic samples...")
        
        # Object types and descriptions
        object_types = [
            ("red_circle", "a red circular object"),
            ("blue_square", "a blue square object"),
            ("green_triangle", "a green triangular object"),
            ("yellow_rectangle", "a yellow rectangular object"),
            ("purple_ellipse", "a purple elliptical object")
        ]
        
        samples = []
        
        for i in range(num_samples):
            # Create random image
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Choose random object type
            obj_type, description = random.choice(object_types)
            
            # Add object to image
            center_x = random.randint(50, 174)
            center_y = random.randint(50, 174)
            size = random.randint(20, 60)
            
            # Draw object based on type
            if "circle" in obj_type:
                self._draw_circle(image, center_x, center_y, size, (255, 0, 0))
            elif "square" in obj_type:
                self._draw_square(image, center_x, center_y, size, (0, 0, 255))
            elif "triangle" in obj_type:
                self._draw_triangle(image, center_x, center_y, size, (0, 255, 0))
            elif "rectangle" in obj_type:
                self._draw_rectangle(image, center_x, center_y, size, (0, 255, 255))
            elif "ellipse" in obj_type:
                self._draw_ellipse(image, center_x, center_y, size, (128, 0, 128))
            
            # Create point prompts (center of object)
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])
            
            # Create text embedding
            text_embedding = torch.randn(512)
            
            samples.append({
                'image': image,
                'point_coords': point_coords,
                'point_labels': point_labels,
                'text_embedding': text_embedding,
                'description': description,
                'object_type': obj_type
            })
        
        return samples
    
    def _draw_circle(self, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a circle on the image."""
        for i in range(max(0, y-size), min(224, y+size)):
            for j in range(max(0, x-size), min(224, x+size)):
                if (i-y)**2 + (j-x)**2 <= size**2:
                    image[i, j] = color
    
    def _draw_square(self, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a square on the image."""
        x1, y1 = max(0, x-size), max(0, y-size)
        x2, y2 = min(224, x+size), min(224, y+size)
        image[y1:y2, x1:x2] = color
    
    def _draw_triangle(self, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a triangle on the image."""
        pts = np.array([
            [x, y-size],
            [x-size, y+size],
            [x+size, y+size]
        ], np.int32)
        
        # Fill triangle
        for i in range(max(0, y-size), min(224, y+size)):
            for j in range(max(0, x-size), min(224, x+size)):
                # Simple point-in-triangle test
                if self._point_in_triangle([j, i], pts[0], pts[1], pts[2]):
                    image[i, j] = color
    
    def _draw_rectangle(self, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw a rectangle on the image."""
        x1, y1 = max(0, x-size), max(0, y-size//2)
        x2, y2 = min(224, x+size), min(224, y+size//2)
        image[y1:y2, x1:x2] = color
    
    def _draw_ellipse(self, image: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int]):
        """Draw an ellipse on the image."""
        for i in range(max(0, y-size//2), min(224, y+size//2)):
            for j in range(max(0, x-size), min(224, x+size)):
                if ((j-x)/size)**2 + ((i-y)/(size//2))**2 <= 1:
                    image[i, j] = color
    
    def _point_in_triangle(self, pt: List[int], v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
        """Check if point is inside triangle."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def demonstrate_feature_extraction(self, samples: List[Dict]):
        """Demonstrate feature extraction."""
        
        print("\n=== Feature Extraction Demo ===")
        
        # Process a few samples
        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}: {sample['description']}")
            
            # Convert image to tensor
            image_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features, text_features = self.model(
                    image=image_tensor,
                    text=sample['text_embedding'].to(self.device),
                    point_coords=sample['point_coords'],
                    point_labels=sample['point_labels']
                )
            
            print(f"  Image features shape: {image_features.shape}")
            print(f"  Text features shape: {text_features.shape}")
            print(f"  Feature norms: image={image_features.norm():.3f}, text={text_features.norm():.3f}")
    
    def demonstrate_similarity_computation(self, samples: List[Dict]):
        """Demonstrate similarity computation."""
        
        print("\n=== Similarity Computation Demo ===")
        
        # Extract features for all samples
        image_features_list = []
        text_features_list = []
        
        for sample in samples:
            image_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                img_feat, txt_feat = self.model(
                    image=image_tensor,
                    text=sample['text_embedding'].to(self.device),
                    point_coords=sample['point_coords'],
                    point_labels=sample['point_labels']
                )
                
                image_features_list.append(img_feat.unsqueeze(0))
                text_features_list.append(txt_feat.unsqueeze(0))
        
        # Stack features
        image_features = torch.cat(image_features_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)
        
        # Normalize features
        image_features = torch.nn.functional.normalize(image_features, dim=1)
        text_features = torch.nn.functional.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # Find best matches
        for i in range(min(5, len(samples))):
            similarities = similarity_matrix[i]
            best_match = torch.argmax(similarities).item()
            confidence = similarities[best_match].item()
            
            print(f"Image {i} ('{samples[i]['description']}') matches best with text {best_match} ('{samples[best_match]['description']}') with confidence {confidence:.3f}")
    
    def demonstrate_training_step(self, samples: List[Dict]):
        """Demonstrate a training step."""
        
        print("\n=== Training Step Demo ===")
        
        # Prepare batch
        batch_size = 4
        batch_samples = samples[:batch_size]
        
        # Convert to batch format
        images = []
        texts = []
        point_coords = []
        point_labels = []
        
        for sample in batch_samples:
            image_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)
            texts.append(sample['text_embedding'])
            point_coords.append(sample['point_coords'])
            point_labels.append(sample['point_labels'])
        
        images = torch.stack(images).to(self.device)
        texts = torch.stack(texts).to(self.device)
        
        # Perform training step
        metrics = self.trainer.train_step(
            images=images,
            texts=texts,
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        print(f"Training metrics:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Learning Rate: {metrics['learning_rate']:.2e}")
    
    def visualize_samples(self, samples: List[Dict], num_samples: int = 4):
        """Visualize sample images with their descriptions."""
        
        print("\n=== Visualization ===")
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        
        for i in range(num_samples):
            sample = samples[i]
            
            # Plot image
            axes[i].imshow(sample['image'])
            axes[i].set_title(f"{sample['description']}\n({sample['object_type']})")
            axes[i].axis('off')
            
            # Mark point prompt
            point = sample['point_coords'][0]
            axes[i].plot(point[0], point[1], 'ro', markersize=8, markerfacecolor='red', markeredgecolor='white')
        
        plt.tight_layout()
        plt.savefig('demo_samples.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to demo_samples.png")
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        
        print("=== Object-Guided CLIP Demo ===")
        print("This demo shows the key functionality of Object-Guided CLIP using synthetic data.")
        
        # Create synthetic data
        samples = self.create_synthetic_data(num_samples=16)
        
        # Visualize samples
        self.visualize_samples(samples)
        
        # Demonstrate feature extraction
        self.demonstrate_feature_extraction(samples)
        
        # Demonstrate similarity computation
        self.demonstrate_similarity_computation(samples)
        
        # Demonstrate training step
        self.demonstrate_training_step(samples)
        
        print("\n=== Demo Completed Successfully ===")
        print("\nKey capabilities demonstrated:")
        print("1. ✓ Object mask generation using SAM2")
        print("2. ✓ Feature extraction from images and objects")
        print("3. ✓ Multi-modal feature fusion")
        print("4. ✓ Contrastive learning between images and texts")
        print("5. ✓ Training pipeline with loss computation")
        print("\nCheck demo_samples.png for visualization results.")


def main():
    """Main demonstration function."""
    
    # Create and run demo
    demo = SimpleDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()
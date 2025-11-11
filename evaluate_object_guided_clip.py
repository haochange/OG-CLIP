# Object-Guided CLIP Evaluation Script
# Evaluation script for Object-Guided Contrastive Language-Image Pre-training

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import os
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from tqdm import tqdm

from object_guided_clip import create_object_guided_clip
from train_object_guided_clip import ObjectGuidedDataset


class ObjectGuidedCLIPEvaluator:
    """Evaluator for Object-Guided CLIP models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run evaluation on
        """
        self.device = device
        
        # Load model
        print("Loading model...")
        self.model = create_object_guided_clip(device=device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def extract_features(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        point_coords: List[np.ndarray],
        point_labels: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image and text features.
        
        Args:
            images: Batch of images [B, C, H, W]
            texts: Batch of text embeddings [B, D]
            point_coords: List of point coordinates
            point_labels: List of point labels
        
        Returns:
            image_features: Extracted image features [B, D]
            text_features: Extracted text features [B, D]
        """
        batch_size = images.size(0)
        all_image_features = []
        all_text_features = []
        
        with torch.no_grad():
            for i in range(batch_size):
                img = images[i]
                txt = texts[i]
                
                img_features, txt_features, _ = self.model(
                    image=img,
                    text=txt,
                    point_coords=point_coords[i],
                    point_labels=point_labels[i]
                )
                
                all_image_features.append(img_features)
                all_text_features.append(txt_features)
        
        image_features = torch.cat(all_image_features, dim=0)
        text_features = torch.cat(all_text_features, dim=0)
        
        return image_features, text_features
    
    def compute_similarity_matrix(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between image and text features.
        
        Args:
            image_features: Image features [B, D]
            text_features: Text features [B, D]
        
        Returns:
            similarity_matrix: Similarity scores [B, B]
        """
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        return similarity_matrix
    
    def evaluate_retrieval(
        self,
        test_dataset: ObjectGuidedDataset,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate image-text retrieval performance.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
        
        Returns:
            metrics: Dictionary containing retrieval metrics
        """
        print("Evaluating retrieval performance...")
        
        from torch.utils.data import DataLoader
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        all_image_features = []
        all_text_features = []
        
        # Extract features
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting features"):
                images = batch['images'].to(self.device)
                texts = batch['text_embeddings'].to(self.device)
                
                image_features, text_features = self.extract_features(
                    images, texts, batch['point_coords'], batch['point_labels']
                )
                
                all_image_features.append(image_features)
                all_text_features.append(text_features)
        
        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(
            all_image_features, all_text_features
        )
        
        # Compute retrieval metrics
        metrics = self._compute_retrieval_metrics(similarity_matrix)
        
        return metrics
    
    def _compute_retrieval_metrics(self, similarity_matrix: torch.Tensor) -> Dict[str, float]:
        """Compute retrieval metrics from similarity matrix."""
        n_samples = similarity_matrix.size(0)
        
        # Image-to-text retrieval
        image_to_text_ranks = []
        for i in range(n_samples):
            similarities = similarity_matrix[i]
            _, indices = torch.sort(similarities, descending=True)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            image_to_text_ranks.append(rank)
        
        # Text-to-image retrieval
        text_to_image_ranks = []
        for i in range(n_samples):
            similarities = similarity_matrix[:, i]
            _, indices = torch.sort(similarities, descending=True)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            text_to_image_ranks.append(rank)
        
        # Compute metrics
        image_to_text_ranks = np.array(image_to_text_ranks)
        text_to_image_ranks = np.array(text_to_image_ranks)
        
        metrics = {
            'image_to_text_r1': np.mean(image_to_text_ranks <= 1),
            'image_to_text_r5': np.mean(image_to_text_ranks <= 5),
            'image_to_text_r10': np.mean(image_to_text_ranks <= 10),
            'image_to_text_median_rank': np.median(image_to_text_ranks),
            'text_to_image_r1': np.mean(text_to_image_ranks <= 1),
            'text_to_image_r5': np.mean(text_to_image_ranks <= 5),
            'text_to_image_r10': np.mean(text_to_image_ranks <= 10),
            'text_to_image_median_rank': np.median(text_to_image_ranks),
            'mean_rank': (np.mean(image_to_text_ranks) + np.mean(text_to_image_ranks)) / 2
        }
        
        return metrics
    
    def evaluate_object_detection(
        self,
        test_dataset: ObjectGuidedDataset,
        batch_size: int = 16
    ) -> Dict[str, float]:
        """
        Evaluate object detection performance using mask quality.
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
        
        Returns:
            metrics: Dictionary containing detection metrics
        """
        print("Evaluating object detection performance...")
        
        from torch.utils.data import DataLoader
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        all_mask_qualities = []
        
        # For synthetic data, we evaluate mask quality based on object center consistency
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating masks"):
                images = batch['images'].to(self.device)
                
                for i in range(images.size(0)):
                    img = images[i]
                    point_coords = batch['point_coords'][i]
                    point_labels = batch['point_labels'][i]
                    
                    # Get mask from model
                    _, _, mask_logits = self.model(
                        image=img,
                        text=torch.randn(512).to(self.device),  # Dummy text
                        point_coords=point_coords,
                        point_labels=point_labels
                    )
                    
                    # Convert mask to binary
                    mask = (mask_logits > 0).float()
                    
                    # For synthetic data, we assume the mask should be centered around the point
                    # In practice, you would compare with ground truth masks
                    mask_quality = self._evaluate_mask_quality(mask, point_coords[0])
                    all_mask_qualities.append(mask_quality)
        
        metrics = {
            'mean_mask_quality': np.mean(all_mask_qualities),
            'std_mask_quality': np.std(all_mask_qualities),
            'min_mask_quality': np.min(all_mask_qualities),
            'max_mask_quality': np.max(all_mask_qualities)
        }
        
        return metrics
    
    def _evaluate_mask_quality(self, mask: torch.Tensor, point_coords: np.ndarray) -> float:
        """Evaluate mask quality based on center consistency."""
        mask_np = mask.cpu().numpy()
        
        # Find mask center
        y_coords, x_coords = np.where(mask_np > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return 0.0
        
        mask_center_x = np.mean(x_coords)
        mask_center_y = np.mean(y_coords)
        
        # Compute distance from expected center (point_coords)
        expected_x, expected_y = point_coords
        distance = np.sqrt((mask_center_x - expected_x)**2 + (mask_center_y - expected_y)**2)
        
        # Convert distance to quality score (inverse relationship)
        max_distance = np.sqrt(mask_np.shape[0]**2 + mask_np.shape[1]**2)
        quality = max(0, 1 - distance / max_distance)
        
        return quality
    
    def _collate_fn(self, batch):
        """Custom collate function."""
        images = torch.stack([torch.from_numpy(item['image']).permute(2, 0, 1).float() / 255.0 
                             for item in batch])
        
        point_coords = [item['point_coords'] for item in batch]
        point_labels = [item['point_labels'] for item in batch]
        text_embeddings = torch.stack([item['text_embedding'] for item in batch])
        descriptions = [item['description'] for item in batch]
        
        return {
            'images': images,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'text_embeddings': text_embeddings,
            'descriptions': descriptions
        }
    
    def visualize_results(
        self,
        test_dataset: ObjectGuidedDataset,
        num_samples: int = 8,
        save_path: str = "evaluation_results.png"
    ):
        """Visualize evaluation results."""
        print("Visualizing results...")
        
        fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
        
        for i in range(num_samples):
            # Get sample
            sample = test_dataset[i]
            
            # Extract features
            with torch.no_grad():
                img_tensor = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                text_tensor = sample['text_embedding'].unsqueeze(0).to(self.device)
                
                image_features, text_features, mask_logits = self.model(
                    image=img_tensor[0],
                    text=text_tensor[0],
                    point_coords=sample['point_coords'],
                    point_labels=sample['point_labels']
                )
                
                # Get mask
                mask = (mask_logits > 0).float().cpu().numpy()
            
            # Plot original image
            axes[0, i].imshow(sample['image'])
            axes[0, i].set_title(f"Original: {sample['description']}")
            axes[0, i].axis('off')
            
            # Plot with mask overlay
            axes[1, i].imshow(sample['image'])
            axes[1, i].imshow(mask, alpha=0.5, cmap='jet')
            axes[1, i].set_title(f"Mask: {sample['description']}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")


def main():
    """Main evaluation function."""
    print("=== Object-Guided CLIP Evaluation ===")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ObjectGuidedDataset(
        data_root=".",
        annotation_file="synthetic_annotations.json",
        split="test"
    )
    
    # Create evaluator
    print("Creating evaluator...")
    evaluator = ObjectGuidedCLIPEvaluator(
        model_path="best_model.pth",
        device=device
    )
    
    # Evaluate retrieval
    print("Evaluating retrieval...")
    retrieval_metrics = evaluator.evaluate_retrieval(test_dataset)
    print("\nRetrieval Metrics:")
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate object detection
    print("\nEvaluating object detection...")
    detection_metrics = evaluator.evaluate_object_detection(test_dataset)
    print("\nObject Detection Metrics:")
    for key, value in detection_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize results
    print("\nVisualizing results...")
    evaluator.visualize_results(test_dataset, num_samples=8)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
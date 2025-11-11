# Object-Guided CLIP Training Script
# Training script for Object-Guided Contrastive Language-Image Pre-training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm

from object_guided_clip import create_object_guided_clip, ObjectGuidedCLIPTrainer


class ObjectGuidedDataset(Dataset):
    """
    Dataset for Object-Guided CLIP training.
    
    This dataset provides images with object annotations and corresponding text descriptions.
    """
    
    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        image_size: int = 224,
        max_objects_per_image: int = 5,
        split: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing images
            annotation_file: JSON file with annotations
            image_size: Target image size for resizing
            max_objects_per_image: Maximum objects to consider per image
            split: Dataset split (train/val/test)
        """
        self.data_root = data_root
        self.image_size = image_size
        self.max_objects_per_image = max_objects_per_image
        self.split = split
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter by split
        self.annotations = [ann for ann in self.annotations if ann.get('split', 'train') == split]
        
        # Create list of (image_path, object_data, text) tuples
        self.samples = []
        for ann in self.annotations:
            image_path = os.path.join(data_root, ann['image_path'])
            
            # Limit number of objects per image
            objects = ann.get('objects', [])[:max_objects_per_image]
            descriptions = ann.get('descriptions', [])[:max_objects_per_image]
            
            for obj_data, desc in zip(objects, descriptions):
                self.samples.append({
                    'image_path': image_path,
                    'object_data': obj_data,
                    'description': desc,
                    'image_id': ann.get('image_id', '')
                })
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Extract object data
        obj_data = sample['object_data']
        
        # Get point prompts (center of bounding box or segmentation mask)
        if 'bbox' in obj_data:
            # Use bounding box center
            x1, y1, x2, y2 = obj_data['bbox']
            point_coords = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
        elif 'segmentation' in obj_data:
            # Use segmentation mask center
            mask = np.array(obj_data['segmentation'])
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                point_coords = np.array([[center_x, center_y]])
            else:
                # Fallback to image center
                point_coords = np.array([[self.image_size // 2, self.image_size // 2]])
        else:
            # Fallback to image center
            point_coords = np.array([[self.image_size // 2, self.image_size // 2]])
        
        # Create point labels (foreground)
        point_labels = np.array([1])
        
        # Create text embedding (in practice, this would come from a text encoder)
        # For this demo, we create a random embedding that represents the text
        text_embedding = torch.randn(512)  # Simulated CLIP text embedding
        
        return {
            'image': image_array,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'text_embedding': text_embedding,
            'description': sample['description'],
            'image_id': sample['image_id']
        }


def create_synthetic_dataset(num_samples: int = 1000, image_size: int = 224) -> ObjectGuidedDataset:
    """
    Create a synthetic dataset for demonstration purposes.
    
    This creates random images with simple geometric objects and corresponding descriptions.
    """
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples, image_size, split="train"):
            self.num_samples = num_samples
            self.image_size = image_size
            self.split = split
            
            # Object types and descriptions
            self.object_types = [
                ("red_circle", "a red circular object"),
                ("blue_square", "a blue square object"),
                ("green_triangle", "a green triangular object"),
                ("yellow_rectangle", "a yellow rectangular object"),
                ("purple_ellipse", "a purple elliptical object")
            ]
            
            print(f"Created synthetic dataset with {num_samples} samples for {split}")
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create random image
            image = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Choose random object type
            obj_type, description = random.choice(self.object_types)
            
            # Add object to image (simplified)
            center_x = random.randint(50, self.image_size - 50)
            center_y = random.randint(50, self.image_size - 50)
            size = random.randint(20, 60)
            
            if "circle" in obj_type:
                cv2.circle(image, (center_x, center_y), size, (255, 0, 0), -1)
            elif "square" in obj_type:
                cv2.rectangle(image, (center_x-size, center_y-size), 
                            (center_x+size, center_y+size), (0, 0, 255), -1)
            elif "triangle" in obj_type:
                pts = np.array([[center_x, center_y-size], 
                              [center_x-size, center_y+size], 
                              [center_x+size, center_y+size]], np.int32)
                cv2.fillPoly(image, [pts], (0, 255, 0))
            elif "rectangle" in obj_type:
                cv2.rectangle(image, (center_x-size, center_y-size//2), 
                            (center_x+size, center_y+size//2), (0, 255, 255), -1)
            elif "ellipse" in obj_type:
                cv2.ellipse(image, (center_x, center_y), (size, size//2), 0, 0, 360, (128, 0, 128), -1)
            
            # Create point coordinates (center of object)
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])
            
            # Create text embedding
            text_embedding = torch.randn(512)
            
            return {
                'image': image,
                'point_coords': point_coords,
                'point_labels': point_labels,
                'text_embedding': text_embedding,
                'description': description,
                'image_id': f"synthetic_{idx}"
            }
    
    return SyntheticDataset(num_samples, image_size)


class ObjectGuidedCLIPTrainingManager:
    """Manager class for training Object-Guided CLIP models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=self._collate_fn
            )
        else:
            self.val_loader = None
        
        # Initialize trainer
        self.trainer = ObjectGuidedCLIPTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def _collate_fn(self, batch):
        """Custom collate function to handle variable-sized data."""
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
        
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(self.device)
            text_embeddings = batch['text_embeddings'].to(self.device)
            
            # Training step
            metrics = self.trainer.train_step(
                images=images,
                texts=text_embeddings,
                point_coords=batch['point_coords'],
                point_labels=batch['point_labels']
            )
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'lr': f"{metrics['learning_rate']:.2e}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {'loss': 0, 'accuracy': 0}
        
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['images'].to(self.device)
                text_embeddings = batch['text_embeddings'].to(self.device)
                
                # Forward pass
                batch_size = images.size(0)
                all_image_features = []
                all_text_features = []
                
                for i in range(batch_size):
                    img = images[i]
                    txt = text_embeddings[i]
                    
                    img_features, txt_features, _ = self.model(
                        image=img,
                        text=txt,
                        point_coords=batch['point_coords'][i],
                        point_labels=batch['point_labels'][i]
                    )
                    
                    all_image_features.append(img_features)
                    all_text_features.append(txt_features)
                
                # Stack features
                image_features = torch.cat(all_image_features, dim=0)
                text_features = torch.cat(all_text_features, dim=0)
                
                # Compute loss
                loss = self.model.contrastive_loss(image_features, text_features)
                
                # Compute accuracy
                logits = torch.matmul(image_features, text_features.T) / self.model.temperature
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == torch.arange(batch_size, device=self.device)).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model for all epochs."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        best_val_accuracy = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(
                self.trainer.scheduler.get_last_lr()[0]
            )
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_checkpoint('best_model.pth')
                print(f"  Saved best model with validation accuracy: {best_val_accuracy:.4f}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict(),
            'history': self.history,
            'epoch': len(self.history['train_loss'])
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")


def main():
    """Main training function."""
    print("=== Object-Guided CLIP Training ===")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_object_guided_clip(device=device)
    
    # Create datasets
    print("Creating datasets...")
    # For demo, we use synthetic dataset. Replace with real dataset in practice.
    train_dataset = create_synthetic_dataset(num_samples=500, image_size=224)
    val_dataset = create_synthetic_dataset(num_samples=100, image_size=224)
    
    # Create training manager
    print("Setting up training...")
    trainer = ObjectGuidedCLIPTrainingManager(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,  # Smaller batch size for demo
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_epochs=5,  # Fewer epochs for demo
        device=device
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train()
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Training completed!")


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate schedule
    axes[1, 0].plot(epochs, history['learning_rate'], 'g-')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Loss vs Accuracy scatter
    axes[1, 1].scatter(history['train_loss'], history['train_accuracy'], 
                        c='blue', label='Train', alpha=0.6)
    axes[1, 1].scatter(history['val_loss'], history['val_accuracy'], 
                        c='red', label='Val', alpha=0.6)
    axes[1, 1].set_title('Loss vs Accuracy')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved to training_history.png")


if __name__ == "__main__":
    import cv2
    main()
# Object-Guided CLIP - Installation and Usage Guide
# Quick start guide for Object-Guided Contrastive Language-Image Pre-training

## Quick Installation

### 1. Install Dependencies
```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn tqdm opencv-python
```

### 2. SAM2 Setup (Optional - for real SAM2 integration)
```bash
# Clone SAM2 repository (if you want to use real SAM2)
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### 3. Download Model Weights (Optional)
- SAM2 checkpoints: [SAM2 Releases](https://github.com/facebookresearch/segment-anything-2/releases)
- Download `sam2_hiera_base_plus.pt` for the base model

## Quick Start

### Basic Usage
```python
from object_guided_clip_final import ObjectGuidedCLIP, preprocess_image
import torch
import numpy as np

# Create model (with dummy components for testing)
model = ObjectGuidedCLIP(
    sam2_model=dummy_sam2,  # Replace with real SAM2
    clip_image_encoder=dummy_clip_encoder,  # Replace with real CLIP
    clip_text_encoder=dummy_clip_text_encoder,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Process an image
image = preprocess_image("path/to/your/image.jpg")
point_coords = np.array([[100, 150]])  # Object center
point_labels = np.array([1])  # Foreground

# Extract features
image_features, text_features = model(
    image=image,
    text=torch.randn(512),  # Your text embedding
    point_coords=point_coords,
    point_labels=point_labels
)
```

### Training
```python
from object_guided_clip_final import ObjectGuidedCLIPTrainer

# Create trainer
trainer = ObjectGuidedCLIPTrainer(
    model=model,
    learning_rate=1e-4,
    weight_decay=1e-4
)

# Training step
metrics = trainer.train_step(
    images=batch_images,
    texts=batch_texts,
    point_coords=batch_point_coords,
    point_labels=batch_point_labels
)
```

## File Structure

```
sam2-main/
├── object_guided_clip_final.py      # Main implementation
├── demo_simple.py                   # Simple demonstration
├── train_object_guided_clip.py      # Training script
├── evaluate_object_guided_clip.py   # Evaluation script
├── README_OBJECT_GUIDED_CLIP.md     # Full documentation
└── demo_samples.png                 # Sample visualization
```

## Key Features

### 1. Object-Guided Representation Learning
- Uses SAM2 to generate object masks
- Combines object and mask features with CLIP features
- Learns object-aware representations

### 2. Contrastive Learning
- Image-text alignment at object level
- Temperature-scaled similarity computation
- Bidirectional loss (image-to-text and text-to-image)

### 3. Flexible Architecture
- Modular design with replaceable components
- Support for different CLIP variants
- Configurable feature dimensions

## Common Use Cases

### Object-Centric Image Retrieval
```python
# Find images containing specific objects
similarities = []
for image in image_database:
    img_features, _ = model(image, text_query, point_prompts)
    similarity = torch.cosine_similarity(img_features, text_features)
    similarities.append(similarity)
```

### Zero-Shot Object Detection
```python
# Detect objects without training data
mask = model.generate_object_mask(image, point_coords, point_labels)
# Use mask for downstream tasks
```

### Multi-Modal Understanding
```python
# Align images and texts at object level
image_features, text_features = model(image, text, point_prompts)
loss = model.contrastive_loss(image_features, text_features)
```

## Performance Tips

### Training
- Use smaller batch sizes (8-16) for better contrastive learning
- Start with lower learning rate (1e-5 to 1e-4)
- Monitor both retrieval and detection metrics

### Inference
- Use multiple point prompts for complex objects
- Combine positive and negative points for better segmentation
- Normalize features before similarity computation

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Poor mask quality**: Try different point prompts or multiple points
3. **Low retrieval accuracy**: Check text embeddings and temperature parameter

### Performance Optimization
- Use mixed precision training for faster computation
- Cache SAM2 features for repeated images
- Batch process multiple objects simultaneously

## Advanced Usage

### Custom Feature Fusion
```python
# Modify feature fusion in the model
class CustomFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Your custom fusion layers
    
    def forward(self, mask_features, object_features, clip_features):
        # Your fusion logic
        return fused_features
```

### Custom Loss Functions
```python
# Implement custom contrastive loss
def custom_contrastive_loss(image_features, text_features, temperature=0.07):
    # Your loss computation
    return loss
```

## Citation

If you use this implementation, please cite:
```bibtex
@misc{object_guided_clip_2024,
  title={Object-Guided Contrastive Language-Image Pre-training},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## Support

For issues and questions:
- Open a GitHub issue
- Check the full documentation in `README_OBJECT_GUIDED_CLIP.md`
- Review the demo scripts for examples
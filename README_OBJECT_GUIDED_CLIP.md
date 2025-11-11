# Object-Guided CLIP - Project Documentation
# Object-Guided Contrastive Language-Image Pre-training Implementation

## Project Overview

This project implements **Object-Guided Contrastive Language-Image Pre-training**, a novel approach that combines SAM2 (Segment Anything Model 2) for object segmentation with CLIP for contrastive learning. The key innovation is using object masks to guide the contrastive learning process, enabling better object-level understanding.

## Architecture

### Core Components

1. **SAM2 Integration**: Uses SAM2 to generate high-quality object masks from point prompts
2. **Dual Path Processing**: Processes both the original image and object masks through separate convolutional layers
3. **Feature Fusion**: Combines mask features, object features, and CLIP image features
4. **Contrastive Learning**: Applies contrastive loss between image and text embeddings

### Model Architecture

```
Input Image + Point Prompts
    ↓
SAM2 → Object Mask
    ↓
Parallel Processing:
├── Object Conv → Object Features
├── Mask Conv → Mask Features  
└── CLIP Image Encoder → Image Features
    ↓
Feature Fusion (Concatenation + Conv)
    ↓
Projection Head → Final Image Embedding
    ↓
Contrastive Loss with Text Embedding
```

## Key Features

### 1. Object-Aware Representation Learning
- Uses SAM2 masks to focus on specific objects
- Separate processing paths for objects and masks
- Better object-level understanding

### 2. Multi-Modal Fusion
- Combines visual features from multiple sources
- Learnable fusion mechanism
- Rich representation learning

### 3. Contrastive Learning
- Image-text alignment
- Temperature-scaled similarity
- Bidirectional loss (image-to-text and text-to-image)

## Implementation Files

### Core Implementation
- **`object_guided_clip_final.py`**: Complete model implementation with all components
- **`object_guided_clip.py`**: Original implementation (legacy)

### Training and Evaluation
- **`train_object_guided_clip.py`**: Complete training pipeline
- **`evaluate_object_guided_clip.py`**: Comprehensive evaluation script
- **`demo_object_guided_clip.py`**: Interactive demonstration

### Usage Examples

#### Basic Usage
```python
from object_guided_clip_final import create_object_guided_clip, preprocess_image

# Create model
model = create_object_guided_clip(
    sam2_config_path="sam2_hiera_b+.yaml",
    sam2_checkpoint_path="sam2_hiera_base_plus.pt",
    device="cuda"
)

# Preprocess image
image_tensor = preprocess_image("path/to/image.jpg")

# Create point prompts (e.g., from bounding box)
point_coords = np.array([[100, 150]])  # Object center
point_labels = np.array([1])  # Foreground point

# Forward pass
image_features, text_features = model(
    image=image_tensor,
    text=text_embedding,
    point_coords=point_coords,
    point_labels=point_labels
)
```

#### Training
```python
from train_object_guided_clip import ObjectGuidedCLIPTrainingManager

# Create training manager
trainer = ObjectGuidedCLIPTrainingManager(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=10
)

# Train model
history = trainer.train()
```

#### Evaluation
```python
from evaluate_object_guided_clip import ObjectGuidedCLIPEvaluator

# Create evaluator
evaluator = ObjectGuidedCLIPEvaluator(
    model_path="best_model.pth",
    device="cuda"
)

# Evaluate retrieval
retrieval_metrics = evaluator.evaluate_retrieval(test_dataset)

# Evaluate object detection
detection_metrics = evaluator.evaluate_object_detection(test_dataset)
```

## Technical Details

### SAM2 Integration
- Uses SAM2ImagePredictor for mask generation
- Supports point prompts, box prompts, and mask prompts
- Multi-mask output with score-based selection

### Feature Processing
- **Mask Convolution**: 2-layer conv network for mask processing
- **Object Convolution**: 2-layer conv network for object processing  
- **CLIP Features**: Extracted from CLIP image encoder
- **Feature Fusion**: 1x1 conv + global average pooling

### Loss Function
```python
def contrastive_loss(self, image_features, text_features):
    # Normalize features
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Bidirectional cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2
```

## Experimental Results

### Retrieval Performance
- **Image-to-Text R@1**: XX.X%
- **Text-to-Image R@1**: XX.X%
- **Mean Rank**: XX.X

### Object Detection
- **Mask Quality**: XX.X%
- **Object Localization**: XX.X%

## Advantages

1. **Object-Level Understanding**: Better captures object semantics
2. **Flexible Prompting**: Supports various prompt types (points, boxes, masks)
3. **Multi-Task Learning**: Combines segmentation and contrastive learning
4. **Transfer Learning**: Leverages pre-trained SAM2 and CLIP models

## Applications

### 1. Object-Centric Retrieval
- Find images containing specific objects
- Object-based image search
- Fine-grained visual understanding

### 2. Zero-Shot Object Detection
- Detect objects without training data
- Transfer learning to new object categories
- Few-shot learning scenarios

### 3. Multi-Modal Understanding
- Image-text alignment at object level
- Visual question answering
- Image captioning with object focus

## Future Work

### 1. Architecture Improvements
- Attention-based feature fusion
- Hierarchical object representation
- Multi-scale feature integration

### 2. Training Enhancements
- Hard negative mining
- Curriculum learning
- Multi-task loss balancing

### 3. Evaluation Extensions
- More comprehensive benchmarks
- Cross-dataset evaluation
- Human evaluation studies

## Requirements

### Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
Pillow>=8.3.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

### SAM2 Installation
```bash
# Clone SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

### Model Weights
- SAM2 checkpoints: Download from [SAM2 releases](https://github.com/facebookresearch/segment-anything-2/releases)
- CLIP models: Automatically downloaded via `torchvision`

## Usage Tips

### 1. Point Prompt Selection
- Use object centers for best results
- Multiple points for complex objects
- Combine positive and negative points

### 2. Training Configuration
- Start with lower learning rate (1e-5 to 1e-4)
- Use cosine annealing scheduler
- Monitor both retrieval and detection metrics

### 3. Evaluation Strategy
- Use multiple random seeds
- Report mean and standard deviation
- Include qualitative results

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{object_guided_clip_2024,
  title={Object-Guided Contrastive Language-Image Pre-training},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-repo/object-guided-clip}}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].
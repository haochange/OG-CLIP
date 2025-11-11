# Object-Guided CLIP Demo Script
# Demonstrates the usage of Object-Guided Contrastive Language-Image Pre-training

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from object_guided_clip_final import create_object_guided_clip, preprocess_image, ObjectGuidedCLIP, ObjectGuidedCLIPTrainer


def create_demo_image_with_objects():
    """Create a demo image with simple geometric objects."""
    # Create a 256x256 RGB image
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add a red square
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
    
    # Add a blue circle
    cv2.circle(image, (180, 180), 30, (0, 0, 255), -1)
    
    # Add a green triangle
    triangle = np.array([[200, 50], [250, 100], [175, 125]], np.int32)
    cv2.fillPoly(image, [triangle], (0, 255, 0))
    
    return image


def demo_object_guided_clip():
    """Demonstrate Object-Guided CLIP functionality."""
    print("=== Object-Guided CLIP Demo ===")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = create_object_guided_clip(device=device)
    print("Model created successfully!")
    
    # Create demo image
    demo_image = create_demo_image_with_objects()
    print("Demo image created with geometric objects")
    
    # Save demo image
    Image.fromarray(demo_image).save("demo_image.png")
    print("Demo image saved as demo_image.png")
    
    # Define point prompts for different objects
    prompts = [
        {
            "name": "Red Square",
            "point_coords": np.array([[100, 100]]),  # Center of red square
            "point_labels": np.array([1]),  # Foreground point
            "text_embedding": torch.randn(1, 512).to(device)  # Simulated text embedding
        },
        {
            "name": "Blue Circle", 
            "point_coords": np.array([[180, 180]]),  # Center of blue circle
            "point_labels": np.array([1]),
            "text_embedding": torch.randn(1, 512).to(device)
        },
        {
            "name": "Green Triangle",
            "point_coords": np.array([[208, 92]]),  # Approximate center of triangle
            "point_labels": np.array([1]),
            "text_embedding": torch.randn(1, 512).to(device)
        }
    ]
    
    print("\nProcessing different objects in the image...")
    
    # Process each object
    results = []
    for prompt in prompts:
        print(f"\nProcessing {prompt['name']}...")
        
        # Extract features using Object-Guided CLIP
        with torch.no_grad():
            image_features, text_features, object_features = model(
                image=demo_image,
                text=prompt['text_embedding'],
                point_coords=prompt['point_coords'],
                point_labels=prompt['point_labels']
            )
        
        results.append({
            "name": prompt['name'],
            "image_features": image_features,
            "text_features": text_features,
            "object_features": object_features,
            "point_coords": prompt['point_coords']
        })
        
        print(f"  Image features shape: {image_features.shape}")
        print(f"  Text features shape: {text_features.shape}")
        print(f"  Object features shape: {object_features.shape}")
    
    # Compute similarities between objects and their corresponding text
    print("\nComputing object-text similarities...")
    for i, result in enumerate(results):
        similarity = torch.cosine_similarity(
            result['image_features'], 
            result['text_features'], 
            dim=-1
        )
        print(f"{result['name']} similarity with text: {similarity.item():.4f}")
    
    # Compute similarities between different objects
    print("\nComputing inter-object similarities...")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            similarity = torch.cosine_similarity(
                results[i]['image_features'],
                results[j]['image_features'],
                dim=-1
            )
            print(f"{results[i]['name']} vs {results[j]['name']}: {similarity.item():.4f}")
    
    # Visualize results
    print("\nVisualizing results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(demo_image)
    axes[0, 0].set_title('Original Demo Image')
    axes[0, 0].axis('off')
    
    # Plot point prompts
    axes[0, 1].imshow(demo_image)
    for result in results:
        point = result['point_coords'][0]
        axes[0, 1].plot(point[0], point[1], 'ro', markersize=10)
        axes[0, 1].annotate(result['name'], (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_title('Object Prompts')
    axes[0, 1].axis('off')
    
    # Feature similarity heatmap
    feature_matrix = torch.zeros(len(results), len(results))
    for i in range(len(results)):
        for j in range(len(results)):
            similarity = torch.cosine_similarity(
                results[i]['image_features'],
                results[j]['image_features'],
                dim=-1
            )
            feature_matrix[i, j] = similarity
    
    im = axes[1, 0].imshow(feature_matrix.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('Feature Similarity Matrix')
    axes[1, 0].set_xticks(range(len(results)))
    axes[1, 0].set_yticks(range(len(results)))
    axes[1, 0].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[1, 0].set_yticklabels([r['name'] for r in results])
    plt.colorbar(im, ax=axes[1, 0])
    
    # Object feature visualization (simplified)
    axes[1, 1].bar(range(len(results)), 
                    [r['image_features'].norm().item() for r in results])
    axes[1, 1].set_title('Feature Magnitudes')
    axes[1, 1].set_xticks(range(len(results)))
    axes[1, 1].set_xticklabels([r['name'] for r in results], rotation=45)
    axes[1, 1].set_ylabel('L2 Norm')
    
    plt.tight_layout()
    plt.savefig('object_guided_clip_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as object_guided_clip_demo.png")
    
    # Test training functionality
    print("\n=== Testing Training Functionality ===")
    trainer = ObjectGuidedCLIPTrainer(model)
    
    # Create dummy batch data
    batch_images = torch.randn(4, 3, 224, 224).to(device)
    batch_texts = torch.randn(4, 512).to(device)
    
    # Single training step
    metrics = trainer.train_step(batch_images, batch_texts)
    print(f"Training metrics: {metrics}")
    
    print("\n=== Demo completed successfully! ===")
    print("Check the generated files:")
    print("- demo_image.png: Original demo image")
    print("- object_guided_clip_demo.png: Visualization results")


def demo_real_image(image_path: str):
    """Demonstrate Object-Guided CLIP on a real image."""
    print(f"\n=== Processing Real Image: {image_path} ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_object_guided_clip(device=device)
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        print(f"Image loaded: {image_array.shape}")
        
        # Define prompts for different regions (you can adjust these)
        height, width = image_array.shape[:2]
        prompts = [
            {
                "name": "Top Left",
                "point_coords": np.array([[width//4, height//4]]),
                "point_labels": np.array([1]),
                "text_embedding": torch.randn(1, 512).to(device)
            },
            {
                "name": "Center",
                "point_coords": np.array([[width//2, height//2]]),
                "point_labels": np.array([1]),
                "text_embedding": torch.randn(1, 512).to(device)
            },
            {
                "name": "Bottom Right",
                "point_coords": np.array([[3*width//4, 3*height//4]]),
                "point_labels": np.array([1]),
                "text_embedding": torch.randn(1, 512).to(device)
            }
        ]
        
        # Process each region
        results = []
        for prompt in prompts:
            print(f"Processing {prompt['name']} region...")
            
            with torch.no_grad():
                image_features, text_features, object_features = model(
                    image=image_array,
                    text=prompt['text_embedding'],
                    point_coords=prompt['point_coords'],
                    point_labels=prompt['point_labels']
                )
            
            results.append({
                "name": prompt['name'],
                "image_features": image_features,
                "text_features": text_features,
                "point_coords": prompt['point_coords']
            })
        
        # Visualize results
        plt.figure(figsize=(10, 8))
        
        # Original image with prompts
        plt.subplot(1, 2, 1)
        plt.imshow(image_array)
        for result in results:
            point = result['point_coords'][0]
            plt.plot(point[0], point[1], 'ro', markersize=8)
            plt.annotate(result['name'], (point[0], point[1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        plt.title('Image with Region Prompts')
        plt.axis('off')
        
        # Feature similarities
        plt.subplot(1, 2, 2)
        feature_matrix = torch.zeros(len(results), len(results))
        for i in range(len(results)):
            for j in range(len(results)):
                similarity = torch.cosine_similarity(
                    results[i]['image_features'],
                    results[j]['image_features'],
                    dim=-1
                )
                feature_matrix[i, j] = similarity
        
        im = plt.imshow(feature_matrix.cpu().numpy(), cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Region Feature Similarities')
        plt.xticks(range(len(results)), [r['name'] for r in results], rotation=45)
        plt.yticks(range(len(results)), [r['name'] for r in results])
        plt.colorbar(im)
        
        plt.tight_layout()
        output_path = f"{image_path.split('.')[0]}_ogclip_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    # Run demo with synthetic image
    demo_object_guided_clip()
    
    # Optionally test with a real image (uncomment and provide path)
    # demo_real_image("path/to/your/image.jpg")
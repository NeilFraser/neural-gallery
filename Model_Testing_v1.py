import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
import glob
from NNDL_Project_Model_v1 import SiameseNetwork




def test_model(model_path='siamese_model.pth', pool_dir='Pool', sample_dir=None, num_samples=5, device=None):
    """
    Test the trained Siamese network on image pairs and visualize results.

    Args:
        model_path (str): Path to the saved model
        pool_dir (str): Directory containing original pool images
        sample_dir (str): Directory containing test samples (if None, uses random pool images)
        num_samples (int): Number of random sample pairs to display
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get all pool images
    pool_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        pool_images.extend(glob.glob(os.path.join(pool_dir, ext)))


    if len(pool_images) == 0:
        print(f"No images found in pool directory: {pool_dir}")
        return

    # If sample directory is provided, use those images, otherwise use random pool images
    test_images = []
    if sample_dir and os.path.exists(sample_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(sample_dir, "**", ext), recursive=True))

    if len(test_images) == 0:
        print("Using random pool images for testing")
        test_images = pool_images

    # Create image pairs for testing
    pairs = []
    num_pairs = min(num_samples, len(test_images))

    # Sample some pairs
    import random
    sample_images = random.sample(test_images, num_pairs)

    # For each sample, compare with itself and with a random other image
    for img_path in sample_images:
        # Compare with itself (should be similar)
        pairs.append((img_path, img_path, "self"))

        # Compare with a random different image (should be dissimilar)
        other_img = random.choice([x for x in pool_images if x != img_path])
        pairs.append((img_path, other_img, "other"))

    # Create a figure for visualization
    fig_size = (15, 5 * len(pairs))
    fig, axes = plt.subplots(len(pairs), 3, figsize=fig_size)

    # Custom colormap for similarity score (red to green)
    cmap = LinearSegmentedColormap.from_list('similarity', ['red', 'yellow', 'green'])

    for i, (img1_path, img2_path, pair_type) in enumerate(pairs):
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            similarity = model(img1_tensor, img2_tensor).item()

        # Display images and similarity
        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Image 1: {os.path.basename(img1_path)}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Image 2: {os.path.basename(img2_path)}")
        axes[i, 1].axis('off')

        # Color background based on similarity score
        axes[i, 2].barh(0, similarity, color=cmap(similarity))
        axes[i, 2].barh(0, 1 - similarity, left=similarity, color='lightgray')
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title(f"Similarity Score: {similarity:.3f}")

        # Add expected similarity text
        expected = "Similar" if pair_type == "self" else "Dissimilar"
        axes[i, 2].text(0.5, -0.2, f"Expected: {expected}",
                        ha='center', transform=axes[i, 2].transAxes)

    plt.tight_layout()
    plt.savefig('similarity_test_results.png')
    plt.show()

    # Test all pool images against each other and report statistics
    print("\nTesting similarity across all pool images...")
    similarity_matrix = np.zeros((len(pool_images), len(pool_images)))

    # Extract group information appropriately
    def get_image_group(path):
        # If image is in a subdirectory like "2a7imb/0.png", use the directory name
        dirname = os.path.basename(os.path.dirname(path))
        if dirname != pool_dir:  # It's in a subdirectory
            return dirname
        else:
            # If it's a direct pool image like "2a7imb.png", use the basename without extension
            return os.path.splitext(os.path.basename(path))[0]


    # Get base names for images (to determine expected similarity)
    base_names = [get_image_group(path) for path in pool_images]

    # Calculate similarity for each pair
    for i, img1_path in enumerate(pool_images):
        img1 = Image.open(img1_path).convert('RGB')
        img1_tensor = transform(img1).unsqueeze(0).to(device)

        for j, img2_path in enumerate(pool_images):
            img2 = Image.open(img2_path).convert('RGB')
            img2_tensor = transform(img2).unsqueeze(0).to(device)

            with torch.no_grad():
                similarity = model(img1_tensor, img2_tensor).item()

            similarity_matrix[i, j] = similarity

    # Calculate accuracy statistics
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(pool_images)):
        for j in range(len(pool_images)):
            # Expected similarity (based on base name)
            expected_similar = base_names[i] == base_names[j]
            # Predicted similarity (threshold at 0.5)
            predicted_similar = similarity_matrix[i, j] > 0.5

            if expected_similar and predicted_similar:
                tp += 1
            elif expected_similar and not predicted_similar:
                fn += 1
            elif not expected_similar and predicted_similar:
                fp += 1
            else:
                tn += 1

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Similarity Score')
    plt.title('Similarity Matrix for Pool Images')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.savefig('similarity_matrix.png')
    plt.show()



if __name__ == "__main__":
    test_model()
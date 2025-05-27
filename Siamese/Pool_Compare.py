import torch
from PIL import Image
import glob
from NNDL_Project_Model_v2 import SiameseNetwork


"""
This script compares each image in the 'Pool' directory with every other image.
None of these images should match each other, so any high similarity score indicates
a false positive.
Tests the accuracy of the ResNet model.
"""

model_path='siamese_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def img_compare(img1_path, img2_path):
    """
    Test the trained Siamese network on image pairs and visualize results.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
    """

    # Load the model.
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess images.
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    img1_tensor = SiameseNetwork.TRANSFORM(img1).unsqueeze(0).to(device)
    img2_tensor = SiameseNetwork.TRANSFORM(img2).unsqueeze(0).to(device)

    # Get model prediction.
    with torch.no_grad():
        similarity = model(img1_tensor, img2_tensor).item()

    return similarity


def pool_compare():
    png_files = glob.glob("../Pool/*.png")
    png_files.sort()

    n = len(png_files)
    for i in range(n):
        print("%s:" % png_files[i])
        for j in range(i + 1, n):
            img1_path = png_files[i]
            img2_path = png_files[j]
            similarity = img_compare(img1_path, img2_path)
            if similarity > 0.75:
                print(f"Similar: {img1_path} vs {img2_path} with similarity {similarity}")


if __name__ == "__main__":
    pool_compare()

import torch
from PIL import Image
import torchvision.transforms as transforms
import glob
from NNDL_Project_Model_v3 import DINOv2EmbeddingExtractor



model_path='dino.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#def pool_compare(model_path='siamese_model.pth', pool_dir='Pool', sample_dir=None, num_samples=5, device=None):

def img_compare(img1_path, img2_path):
    """
    Test the trained Siamese network on image pairs and visualize results.

    Args:
        sample_dir (str): Directory containing test samples (if None, uses random pool images)
        num_samples (int): Number of random sample pairs to display
    """

    # Load the model
    model = DINOv2EmbeddingExtractor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess images
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    img1_tensor = DINOv2EmbeddingExtractor.TRANSFORM(img1).unsqueeze(0).to(device)
    img2_tensor = DINOv2EmbeddingExtractor.TRANSFORM(img2).unsqueeze(0).to(device)

    # Get model prediction
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

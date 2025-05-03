import random
import torch
from PIL import Image
from SiameseNetwork import SiameseNetwork


def test_model(img_dir='Test'):
    """
    Test the trained Siamese network on image pairs.

    Args:
        img_dir (str): Directory containing test images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(SiameseNetwork.FILENAME, map_location=device))
    model.to(device)
    model.eval()

    pairs = SiameseNetwork.get_pairs(img_dir)

    errors = 0.0
    count = 0
    for (img1_path, img2_path, expected) in pairs:
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = SiameseNetwork.TRANSFORM(img1).unsqueeze(0).to(device)
        img2_tensor = SiameseNetwork.TRANSFORM(img2).unsqueeze(0).to(device)
        if random.random() > 0.5:
           img1_tensor, img2_tensor = img2_tensor, img1_tensor

        # Get model prediction
        with torch.no_grad():
            similarity = model(img1_tensor, img2_tensor).item()
        print("%s vs %s:\tExpect %f\tActual %f" % (img1_path, img2_path, expected, similarity))

        errors += abs(expected - similarity)
        count += 1

    print()
    print("Average error: %f%%" % (errors / count * 100))


if __name__ == "__main__":
    test_model()

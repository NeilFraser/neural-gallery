import random
import torch
from PIL import Image
from DINOv2 import DINOv2EmbeddingExtractor


""" This script tests the accuracy of a trained DINO network model.
It uses the images in the 'Test' directory, which were never been seen by the
model during training.
Prints statistics about the model's performance.
"""

def test_model(img_dir):
    """
    Test the trained Siamese network on image pairs.

    Args:
        img_dir (str): Directory containing test images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model.
    model = DINOv2EmbeddingExtractor()
    model.load_state_dict(torch.load(DINOv2EmbeddingExtractor.FILENAME, map_location=device))
    model.to(device)
    model.eval()

    pairs = DINOv2EmbeddingExtractor.get_pairs(img_dir)

    worst_false_positive = 0.0
    worst_false_negative = 0.0
    worst_false_positive_pair = None
    worst_false_negative_pair = None

    count = 0
    errors = 0.0
    positive_errors = 0.0
    positive_count = 0
    negative_errors = 0.0
    negative_count = 0
    for (img1_path, img2_path, expected) in pairs:
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = DINOv2EmbeddingExtractor.TRANSFORM(img1).unsqueeze(0).to(device)
        img2_tensor = DINOv2EmbeddingExtractor.TRANSFORM(img2).unsqueeze(0).to(device)
        if random.random() > 0.5:
           img1_tensor, img2_tensor = img2_tensor, img1_tensor

        # Get model prediction
        with torch.no_grad():
            similarity = model(img1_tensor, img2_tensor).item()
        print("%s vs %s:\tExpect %f\tActual %f" % (img1_path, img2_path, expected, similarity))

        loss = abs(expected - similarity)
        if expected < similarity and loss > worst_false_positive:
            worst_false_positive = loss
            worst_false_positive_pair = (img1_path, img2_path)
        if expected > similarity and loss > worst_false_negative:
            worst_false_negative = loss
            worst_false_negative_pair = (img1_path, img2_path)

        errors += loss
        count += 1
        if expected < similarity:
            positive_errors += loss
            positive_count += 1
        else:
            negative_errors += loss
            negative_count += 1

    print()
    print("Average error: %f%%" % (errors / count * 100))
    print("Average overly-positive error: %f%%" % (positive_errors / positive_count * 100))
    print("Average overly-negative error: %f%%" % (negative_errors / negative_count * 100))
    print("Worst false positive: %d%% (%s vs %s)" % (round(worst_false_positive * 100), worst_false_positive_pair[0], worst_false_positive_pair[1]))
    print("Worst false negative: %d%% (%s vs %s)" % (round(worst_false_negative * 100), worst_false_negative_pair[0], worst_false_negative_pair[1]))


if __name__ == "__main__":
    test_model("Test")

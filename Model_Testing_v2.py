import glob
import os
import random
import torch
from PIL import Image
from SiameseNetwork import SiameseNetwork


def get_image_name(image_path):
  base_name = os.path.basename(image_path)
  return os.path.splitext(base_name)[0]


def test_model(img_dir='Validation'):
    """
    Test the trained Siamese network on image pairs.

    Args:
        img_dir (str): Directory containing validation images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(SiameseNetwork.FILENAME, map_location=device))
    model.to(device)
    model.eval()

    # Get all validation images
    png_files = glob.glob(os.path.join(img_dir, "*.png"))
    png_files.sort()
    print("Found %d %s images." % (len(png_files), img_dir))

    # Create image pairs for testing.
    pairs = []
    for png_file in png_files:
      # One pair is the validation image against itself (must match).
      pairs.append((png_file, png_file, SiameseNetwork.SCORE_MATCH))
      # Half is each validation image against its isotopes (should match).
      png_name = get_image_name(png_file)
      isotopes = glob.glob(os.path.join(img_dir, png_name, "*.png"))
      for isotope in isotopes:
         pairs.append((png_file, isotope, SiameseNetwork.SCORE_ISOTOPE))
      # Half is each validation image against each other (should not match).
      non_match_imgs = png_files[:]
      non_match_imgs.remove(png_file)
      non_match_imgs = random.sample(non_match_imgs, min(len(isotopes) + 1, len(non_match_imgs)))
      for non_match in non_match_imgs:
         pairs.append((png_file, non_match, SiameseNetwork.SCORE_NONE))

    errors = 0.0
    count = 0
    for (img1_path, img2_path, expected) in pairs:
        # Load and preprocess images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = SiameseNetwork.TRANSFORM(img1).unsqueeze(0).to(device)
        img2_tensor = SiameseNetwork.TRANSFORM(img2).unsqueeze(0).to(device)

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

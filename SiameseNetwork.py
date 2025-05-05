import glob
import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision import models


class SiameseNetwork(nn.Module):
    FILENAME = "siamese_model.pth"

    SCORE_MATCH = 1.0
    SCORE_ISOTOPE = 0.8
    SCORE_NONE = 0.0

    # Define transformations
    TRANSFORM = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Use pretrained ResNet for feature extraction
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension for ResNet18 is 512
        self.fc1 = nn.Linear(512 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)


    def forward(self, img1, img2):
        # Extract features from both images
        features1 = self.feature_extractor(img1).flatten(1)
        features2 = self.feature_extractor(img2).flatten(1)

        # Concatenate features
        combined_features = torch.cat((features1, features2), dim=1)

        # Predict similarity
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x


    def get_image_name(image_path):
        base_name = os.path.basename(image_path)
        return os.path.splitext(base_name)[0]


    def get_pairs(dir):
        # Check if directory exists
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Directory not found: {dir}")

        # Get all master images
        png_files = glob.glob(os.path.join(dir, "*.png"))
        png_files.sort()
        print("Found %d %s images." % (len(png_files), dir))

        # Create image pairs for comparison.
        pairs = []
        for png_file in png_files:
            # One pair is the test image against itself (must match).
            pairs.append((png_file, png_file, SiameseNetwork.SCORE_MATCH))

            # Half is the test image against all its isotopes (should match).
            png_name = SiameseNetwork.get_image_name(png_file)
            isotopes = glob.glob(os.path.join(dir, png_name, "*.png"))
            for isotope in isotopes:
                pairs.append((png_file, isotope, SiameseNetwork.SCORE_ISOTOPE))

            # Half is the test image an isotope of other test images (should not match).
            non_match_imgs = png_files[:]
            non_match_imgs.remove(png_file)  # Remove itself from non-matching images.
            non_match_imgs = random.sample(non_match_imgs, min(len(isotopes) + 1, len(non_match_imgs)))
            for non_match in non_match_imgs:
                png_name = SiameseNetwork.get_image_name(non_match)
                isotopes = glob.glob(os.path.join(png_name, "*.png"))
                isotopes.append(non_match)  # Add the original to the isotopes.
                non_match_image = random.choice(isotopes)
                pairs.append((png_file, non_match_image, SiameseNetwork.SCORE_NONE))
        # Shuffle the pairs to ensure randomness
        random.shuffle(pairs)
        # Return the pairs
        return pairs

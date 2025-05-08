import glob
import os
import random
import re
import torch
import timm
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision import models


class DINOv2EmbeddingExtractor(nn.Module):
    FILENAME = "dino.pth"

    SCORE_MATCH = 1.0
    SCORE_ISOTOPE = 0.6
    SCORE_NONE = 0.0

    # Define transformations, updated to match DINOv2 input requirements
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self):
        super(DINOv2EmbeddingExtractor, self).__init__()
        # Load pre-trained DINOv2 model
        self.feature_extractor = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
        self.feature_extractor.reset_classifier(0)  # Remove classification head

    def forward(self, img1, img2):
        # Ensure inputs are batched
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)

        # Extract embeddings for both images
        embedding1 = self.feature_extractor(img1)
        embedding2 = self.feature_extractor(img2)

        # Compute cosine similarity between the two embeddings
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        return similarity


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
            pairs.append((png_file, png_file, DINOv2EmbeddingExtractor.SCORE_MATCH))

            # Half is the test image against all its isotopes (should match).
            png_name = DINOv2EmbeddingExtractor.get_image_name(png_file)
            isotopes = glob.glob(os.path.join(dir, png_name, "*.png"))
            for isotope in isotopes:
                # Get the isotope's mutation score from its filename.
                # Example filename: 1_50.png -> 50
                m = re.search(r"\d+_(\d+).png$", isotope)
                score = int(m.group(1)) / 100.0
                # Calculate an expected score based on the mutation.
                score = 1 - (1 - DINOv2EmbeddingExtractor.SCORE_ISOTOPE) * score
                pairs.append((png_file, isotope, score))

            # Half is the test image an isotope of other test images (should not match).
            non_match_imgs = png_files[:]
            non_match_imgs.remove(png_file)  # Remove itself from non-matching images.
            non_match_imgs = random.sample(non_match_imgs, min(len(isotopes) + 1, len(non_match_imgs)))
            for non_match in non_match_imgs:
                png_name = DINOv2EmbeddingExtractor.get_image_name(non_match)
                isotopes = glob.glob(os.path.join(png_name, "*.png"))
                isotopes.append(non_match)  # Add the original to the isotopes.
                non_match_image = random.choice(isotopes)
                pairs.append((png_file, non_match_image, DINOv2EmbeddingExtractor.SCORE_NONE))
        # Shuffle the pairs to ensure randomness
        random.shuffle(pairs)
        # Return the pairs
        return pairs

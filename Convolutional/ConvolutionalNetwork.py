import glob
import os
import random
import re
import torchvision.transforms as transforms
import torch.nn as nn


class ConvolutionalNetwork(nn.Module):
    """
    A Convolutional Neural Network.

    Input: A 200x400 image.
    Output: A single sigmoid value.
    """

    FILENAME = "convolutional_model.pth"

    SCORE_MATCH = 1.0
    SCORE_ISOTOPE = 0.6
    SCORE_NONE = 0.0

    # Transform images to 200x400 and normalize.
    TRANSFORM = transforms.Compose([
        transforms.Resize((200, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, input_channels=3):
        """
        Initializes the layers of the CNN.

        Args:
            input_channels (int): Number of channels in the input image.
                                  1 for grayscale, 3 for RGB.
        """
        super(ConvolutionalNetwork, self).__init__()

        # Convolutional Block 1
        # Input: (Batch, input_channels, 200, 400)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Output shape: (Batch, 16, 200, 400) after conv
        # Formula: ((W-K+2P)/S) + 1 => ((400-5+2*2)/1)+1 = 400; ((200-5+2*2)/1)+1 = 200
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        # Output shape: (Batch, 16, 50, 100) after pool (200/4=50, 400/4=100)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output shape: (Batch, 32, 50, 100) after conv
        # Formula: ((100-3+2*1)/1)+1 = 100; ((50-3+2*1)/1)+1 = 50
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch, 32, 25, 50) after pool (50/2=25, 100/2=50)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Output shape: (Batch, 64, 25, 50) after conv
        # Formula: ((50-3+2*1)/1)+1 = 50; ((25-3+2*1)/1)+1 = 25
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(5,5), stride=(5,5)) # Using tuple for potentially non-square kernel/stride
        # Output shape: (Batch, 64, 5, 10) after pool (25/5=5, 50/5=10)

        # Flatten the output from conv blocks.
        # Calculated flattened features: 64 * 5 * 10 = 3200
        self.fc1_input_features = 64 * 5 * 10

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc1_input_features, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Optional dropout for regularization.
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor (batch_size, 1) with sigmoid activation.
        """
        # Convolutional Block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Convolutional Block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Convolutional Block 3
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the tensor for the fully connected layers.
        # x.view(-1, num_features) where -1 infers the batch size.
        x = x.view(-1, self.fc1_input_features)

        # Fully Connected Layers
        x = self.relu4(self.fc1(x))
        x = self.dropout(x) # Apply dropout.
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


    def get_image_name(image_path):
        base_name = os.path.basename(image_path)
        return os.path.splitext(base_name)[0]


    def get_pairs(dir):
        # Check if directory exists.
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Directory not found: {dir}")

        # Get all master images.
        png_files = glob.glob(os.path.join(dir, "*.png"))
        png_files.sort()
        print("Found %d %s images." % (len(png_files), dir))

        # Create image pairs for comparison.
        pairs = []
        for png_file in png_files:
            # One pair is the test image against itself (must match).
            pairs.append((png_file, png_file, ConvolutionalNetwork.SCORE_MATCH))

            # Half is the test image against all its isotopes (should match).
            png_name = ConvolutionalNetwork.get_image_name(png_file)
            isotopes = glob.glob(os.path.join(dir, png_name, "*.png"))
            for isotope in isotopes:
                # Get the isotope's mutation score from its filename.
                # Example filename: 1_50.png -> 50
                m = re.search(r"\d+_(\d+).png$", isotope)
                score = int(m.group(1)) / 100.0
                # Calculate an expected score based on the mutation.
                score = 1 - (1 - ConvolutionalNetwork.SCORE_ISOTOPE) * score
                pairs.append((png_file, isotope, score))

            # Half is the test image an isotope of other test images (should not match).
            non_match_imgs = png_files[:]
            non_match_imgs.remove(png_file)  # Remove itself from non-matching images.
            non_match_imgs = random.sample(non_match_imgs, min(len(isotopes) + 1, len(non_match_imgs)))
            for non_match in non_match_imgs:
                png_name = ConvolutionalNetwork.get_image_name(non_match)
                isotopes = glob.glob(os.path.join(png_name, "*.png"))
                isotopes.append(non_match)  # Add the original to the isotopes.
                non_match_image = random.choice(isotopes)
                pairs.append((png_file, non_match_image, ConvolutionalNetwork.SCORE_NONE))
        # Shuffle the pairs to ensure randomness.
        random.shuffle(pairs)
        # Return the pairs.
        return pairs

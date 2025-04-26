import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision import models


class SiameseNetwork(nn.Module):
    FILENAME = "siamese_model.pth"

    SCORE_MATCH = 1.0
    SCORE_ISOTOPE = 0.9
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

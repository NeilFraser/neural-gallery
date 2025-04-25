import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image

class SiameseNetwork(nn.Module):
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


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = []

        # Check if directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # Group images by their ID (which is the subdirectory name)
        self.image_groups = {}

        # Traverse subdirectories (each named after an image ID)
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        if not subdirs:
            raise ValueError(f"No subdirectories found in {root_dir}")

        print(f"Found {len(subdirs)} image groups in {root_dir}")

        # Process each subdirectory
        for subdir in subdirs:
            base_name = subdir  # The ID of the image group
            subdir_path = os.path.join(root_dir, subdir)

            # Find all images in this subdirectory
            image_files = [f for f in os.listdir(subdir_path)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"Warning: No images found in {subdir_path}")
                continue

            if base_name not in self.image_groups:
                self.image_groups[base_name] = []

            # Add all images from this subdirectory to the group
            for img_file in image_files:
                full_path = os.path.join(subdir_path, img_file)
                self.all_images.append(full_path)
                self.image_groups[base_name].append(full_path)

        if not self.image_groups:
            raise ValueError("No valid image groups found")

        print(f"Processed {len(self.all_images)} total images across {len(self.image_groups)} groups")

        # Create pairs for training
        self.pairs = []
        self.labels = []

        # Rest of the pairing logic remains the same
        # Positive pairs (image with itself and its variations within same group)
        for base_name, paths in self.image_groups.items():
            for i, path1 in enumerate(paths):
                # Self match
                self.pairs.append((path1, path1))
                self.labels.append(1.0)  # Perfect match

                # Variation matches within same group
                for j, path2 in enumerate(paths):
                    if i != j:
                        self.pairs.append((path1, path2))
                        self.labels.append(1.0)

        # Negative pairs (image with random images from other groups)
        for base_name, paths in self.image_groups.items():
            other_base_names = list(self.image_groups.keys())
            other_base_names.remove(base_name)

            if not other_base_names:  # Skip if there's only one group
                continue

            for path1 in paths:
                for _ in range(17):  # 17 random other images
                    random_base = random.choice(other_base_names)
                    random_path = random.choice(self.image_groups[random_base])
                    self.pairs.append((path1, random_path))
                    self.labels.append(0.0)  # No match

        print(f"Created {len(self.pairs)} image pairs for training/validation")

    # Rest of the class remains the same
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        # Extract group names from paths
        img1_group = os.path.basename(os.path.dirname(img1_path))
        img2_group = os.path.basename(os.path.dirname(img2_path))

        # Combine group and filename
        img1_id = f"{img1_group}/{os.path.basename(img1_path)}"
        img2_id = f"{img2_group}/{os.path.basename(img2_path)}"

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1,
            'img2': img2,
            'label': torch.tensor(label, dtype=torch.float),
            'filename1': img1_id,
            'filename2': img2_id
        }


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

        # Counters for debugging
        self.self_matches = 0
        self.isotope_matches = 0
        self.non_matches = 0
        self.unclassified = 0


    def forward(self, outputs, targets, filename1, filename2):
        batch_size = outputs.size(0)
        device = outputs.device

        # Initialize loss tensors
        self_match_loss = torch.zeros(batch_size, device=device)
        isotope_match_loss = torch.zeros(batch_size, device=device)
        non_match_loss = torch.zeros(batch_size, device=device)

        batch_self = batch_isotope = batch_non = 0

        for i in range(batch_size):
            fname1 = filename1[i]
            fname2 = filename2[i]
            output = outputs[i].squeeze()

            # Extract group and image name
            group1, img1 = fname1.split('/', 1)
            group2, img2 = fname2.split('/', 1)

            is_self_match = (fname1 == fname2)
            is_same_group = (group1 == group2)

            if is_self_match:
                self_match_loss[i] = torch.pow(1 - output, 4)
                batch_self += 1
                self.self_matches += 1
            elif is_same_group:
                isotope_match_loss[i] = 1 - output
                batch_isotope += 1
                self.isotope_matches += 1
            else:
                non_match_loss[i] = output
                batch_non += 1
                self.non_matches += 1

        # Print statistics occasionally
        if random.random() < 0.05:
            print(f"Batch match types: self={batch_self}, isotope={batch_isotope}, non={batch_non}")
            print(f"Total matches: self={self.self_matches}, isotope={self.isotope_matches}, non={self.non_matches}")

        total_loss = self_match_loss.sum() + isotope_match_loss.sum() + non_match_loss.sum()
        return total_loss / batch_size if batch_size > 0 else total_loss

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device='cuda'):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            labels = batch['label'].to(device)
            filename1 = batch['filename1']
            filename2 = batch['filename2']

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img1, img2)

            # Calculate loss
            loss = criterion(outputs, labels, filename1, filename2)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/10:.3f}')
                running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                img1 = batch['img1'].to(device)
                img2 = batch['img2'].to(device)
                labels = batch['label'].to(device)
                filename1 = batch['filename1']
                filename2 = batch['filename2']

                outputs = model(img1, img2)
                loss = criterion(outputs, labels, filename1, filename2)

                val_loss += loss.item()

                # Calculate accuracy (threshold at 0.5)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)

                # Debug information
                print(f"Batch - outputs: {outputs[:5].cpu().numpy()}")
                print(f"Batch - labels: {labels[:5].cpu().numpy()}")
                print(f"Batch - predicted: {predicted[:5].cpu().numpy()}")

                correct += (predicted.view(-1) == labels.view(-1)).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total

        print(f"Total samples: {total}, Correct predictions: {correct}")
        print(f'Epoch {epoch+1} validation loss: {epoch_val_loss:.3f}, accuracy: {epoch_val_acc:.2f}%')

    return model


def test_model_output(model, device):
    model = model.to(device)
    model.eval()
    # Create random test data
    img1 = torch.randn(1, 3, 200, 200).to(device)
    img2 = torch.randn(1, 3, 200, 200).to(device)

    # Test identical images
    with torch.no_grad():
        # Different random images
        output_diff = model(img1, img2).item()
        # Same image
        output_same = model(img1, img1).item()

    print(f"Model output test:")
    print(f"Different images: {output_diff:.4f}")
    print(f"Same image: {output_same:.4f}")
    print(f"Expected: Different < Same")

    return output_diff < output_same

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImagePairDataset(root_dir='Training', transform=transform)
    val_dataset = ImagePairDataset(root_dir='Validation', transform=transform)

    # Create data loaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = SiameseNetwork()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CustomLoss()

    # Test model before training
    print("Testing model before training:")
    test_model_output(model, device)

    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=10,
        device=device
    )

    # Test model after training
    print("Testing model after training:")
    test_model_output(trained_model, device)

    # Save model
    torch.save(trained_model.state_dict(), 'siamese_model.pth')

if __name__ == '__main__':
    main()
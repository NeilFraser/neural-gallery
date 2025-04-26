import glob
import os
import random
import torch
from PIL import Image
from SiameseNetwork import SiameseNetwork
from torch import nn
from torch.utils.data import Dataset, DataLoader


def get_image_name(image_path):
  base_name = os.path.basename(image_path)
  return os.path.splitext(base_name)[0]


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.all_images = []

        # Check if directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # Get all images
        png_files = glob.glob(os.path.join(root_dir, "*.png"))
        png_files.sort()
        print("Found %d %s images." % (len(png_files), root_dir))

        # Create image pairs for testing.
        # Create pairs for training
        self.pairs = []
        for png_file in png_files:
            # One pair is the image against itself (must match).
            self.pairs.append((png_file, png_file, SiameseNetwork.SCORE_MATCH))
            # Half is each image against its isotopes (should match).
            png_name = get_image_name(png_file)
            isotopes = glob.glob(os.path.join(root_dir, png_name, "*.png"))
            for isotope in isotopes:
                self.pairs.append((png_file, isotope, SiameseNetwork.SCORE_ISOTOPE))
            # Half is each image against each other (should not match).
            non_match_imgs = png_files[:]
            non_match_imgs.remove(png_file)
            non_match_imgs = random.sample(non_match_imgs, min(len(isotopes) + 1, len(non_match_imgs)))
            for non_match in non_match_imgs:
                self.pairs.append((png_file, non_match, SiameseNetwork.SCORE_NONE))
      
        print(f"Created {len(self.pairs)} image pairs for training/validation")

    # Rest of the class remains the same
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1,
            'img2': img2,
            'label': torch.tensor(label, dtype=torch.float),
            'filename1': img1_path,
            'filename2': img2_path
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
            dir1, img1 = os.path.split(fname1)
            dir2, img2 = os.path.split(fname2)

            # 'Training/zjfjan.png' and 'Training/zjfjan.png' is a self match
            is_self_match = (fname1 == fname2)
            # 'Training/' vs 'Training/zjfjan' is a source image and an isotope.
            # 'Training/' vs 'Training/' is two different source images.
            is_same_group = (dir1 != dir2)

            if is_self_match:
                self_match_loss[i] = abs(SiameseNetwork.SCORE_MATCH - output)
                batch_self += 1
                self.self_matches += 1
            elif is_same_group:
                isotope_match_loss[i] = abs(SiameseNetwork.SCORE_ISOTOPE - output)
                batch_isotope += 1
                self.isotope_matches += 1
            else:
                non_match_loss[i] = abs(SiameseNetwork.SCORE_NONE - output)
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


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_dataset = ImagePairDataset(root_dir='Training', transform=SiameseNetwork.TRANSFORM)
    val_dataset = ImagePairDataset(root_dir='Validation', transform=SiameseNetwork.TRANSFORM)

    # Create data loaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = SiameseNetwork()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CustomLoss()

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

    # Save model
    torch.save(trained_model.state_dict(), SiameseNetwork.FILENAME)

if __name__ == '__main__':
    main()

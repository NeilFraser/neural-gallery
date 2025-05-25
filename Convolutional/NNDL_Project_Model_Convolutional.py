import random
import torch
from PIL import Image
from ConvolutionalNetwork import ConvolutionalNetwork
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.all_images = []

        self.pairs = ConvolutionalNetwork.get_pairs(root_dir)

        print(f"Created {len(self.pairs)} image pairs")

    # Rest of the class remains the same
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        img1_path, img2_path, target = self.pairs[idx]
        if random.random() > 0.5:
           img1_path, img2_path = img2_path, img1_path

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # Merge images horizontally.
        merged_image = self.merge_two_images_horizontally(img1, img2)

        if self.transform:
            merged_image = self.transform(merged_image)

        return {
            'img': merged_image,
            'target': torch.tensor(target, dtype=torch.float),
        }


    def merge_two_images_horizontally(self, image1, image2):
        """
        Merges two PIL Image objects horizontally.

        Args:
            image1 (PIL.Image.Image): The first image (left side).
            image2 (PIL.Image.Image): The second image (right side).

        Returns:
            PIL.Image.Image: The new merged image, or None if dimensions don't match height.
        """
        if image1.height != image2.height:
            print("Error: Heights of the two images must be the same to merge horizontally.")
            return None

        width1, height1 = image1.size
        width2, height2 = image2.size

        # New image dimensions
        new_width = width1 + width2
        new_height = height1 # or height2, since they are the same

        # Create a new image with the combined width and original height
        merged_image = Image.new(image1.mode, (new_width, new_height))

        # Paste the first image onto the new image
        merged_image.paste(image1, (0, 0))

        # Paste the second image onto the new image, to the right of the first one
        merged_image.paste(image2, (width1, 0))

        return merged_image

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()


    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        device = outputs.device

        # Initialize loss tensors
        loss = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            target = targets[i]
            output = outputs[i].squeeze()
            loss[i] = abs(target - output)

        total_loss = loss.sum()
        return total_loss / batch_size if batch_size > 0 else total_loss


def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device='cuda'):
    model.to(device)

    best_val_loss = 1.0
    for epoch in range(num_epochs):
        model.train()
        #running_loss = 0.0

        for i, batch in enumerate(train_loader):
            img = batch['img'].to(device)
            targets = batch['target'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            #running_loss += loss.item()

            #if i % 10 == 9:
            #    print(f'[{epoch+1}, {i+1}] loss: {running_loss/10:.3f}')
            #    running_loss = 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                img = batch['img'].to(device)
                targets = batch['target'].to(device)

                outputs = model(img)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                total += targets.size(0)

                # Debug information
                #print(f"Batch - outputs: {outputs[:5].cpu().numpy()}")
                #print(f"Batch - targets: {targets[:5].cpu().numpy()}")
                #print(f"Batch - predicted: {predicted[:5].cpu().numpy()}")

        epoch_val_loss = val_loss / len(val_loader)

        print(f"Total samples: {total}, Epoch {epoch+1} validation loss: {epoch_val_loss:.3f}")
        # Append to log file
        with open('training_log.csv', 'a') as log_file:
            log_file.write(f"{epoch+1}, {epoch_val_loss:.3f}\n")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the model if validation loss decreases
            torch.save(model.state_dict(), ConvolutionalNetwork.FILENAME)
            print(f"Model saved with validation loss: {best_val_loss:.3f}")

    return model


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_dataset = ImagePairDataset(root_dir='../Training', transform=ConvolutionalNetwork.TRANSFORM)
    val_dataset = ImagePairDataset(root_dir='../Validation', transform=ConvolutionalNetwork.TRANSFORM)

    # Create data loaders with fewer workers
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = ConvolutionalNetwork()

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
        num_epochs=2000,
        device=device
    )

    # Save model
    #torch.save(trained_model.state_dict(), ConvolutionalNetwork.FILENAME)

if __name__ == '__main__':
    main()

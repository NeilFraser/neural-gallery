import random
import torch
from PIL import Image
from SiameseNetwork import SiameseNetwork
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.all_images = []

        self.pairs = SiameseNetwork.get_pairs(root_dir)

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

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {
            'img1': img1,
            'img2': img2,
            'target': torch.tensor(target, dtype=torch.float),
        }


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
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            targets = batch['target'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(img1, img2)

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
                img1 = batch['img1'].to(device)
                img2 = batch['img2'].to(device)
                targets = batch['target'].to(device)

                outputs = model(img1, img2)
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
        #with open('resnet_training_log.csv', 'a') as log_file:
        #    log_file.write(f"{epoch+1}, {epoch_val_loss:.3f}\n")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the model if validation loss decreases
            torch.save(model.state_dict(), SiameseNetwork.FILENAME)
            print(f"Model saved with validation loss: {best_val_loss:.3f}")

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
        num_epochs=1000,
        device=device
    )

    # Save model
    #torch.save(trained_model.state_dict(), SiameseNetwork.FILENAME)

if __name__ == '__main__':
    main()

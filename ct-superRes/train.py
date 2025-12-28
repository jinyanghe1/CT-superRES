import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SRResNet
from dataset import CTDataset
from tqdm import tqdm

def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset and DataLoader
    train_dataset = CTDataset(args.data_dir, scale_factor=args.scale_factor, crop_size=args.crop_size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Model
    model = SRResNet(in_channels=1, out_channels=1, scale_factor=args.scale_factor).to(device)
    
    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []

    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for lr_imgs, hr_imgs in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{args.epochs}")

                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                # Forward pass
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Plot Loss Curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, args.epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_loss.png'))
    print(f"Saved training loss plot to {args.checkpoint_dir}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_final.pth"))
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT Super Resolution Training')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing training images')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale_factor', type=int, default=4, help='Upsampling scale factor')
    parser.add_argument('--crop_size', type=int, default=128, help='Crop size for HR images')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint save interval (epochs)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')

    args = parser.parse_args()
    train(args)

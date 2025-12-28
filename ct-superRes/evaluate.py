import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SRResNet
from dataset import CTDataset
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).squeeze(0)
    return img

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)

    # Dataset
    test_dataset = CTDataset(args.data_dir, scale_factor=args.scale_factor, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = SRResNet(scale_factor=args.scale_factor).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model not found at {args.model_path}")
        return

    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    total_time = 0
    
    count = 0
    
    # Store results for plotting
    results_metrics = {'PSNR': [], 'SSIM': [], 'MSE': []}

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(tqdm(test_loader)):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device) # Keep on GPU for model, move to CPU for metrics

            start_time = time.time()
            sr_img = model(lr_img)
            end_time = time.time()
            total_time += (end_time - start_time)

            # Convert to numpy for metrics
            sr_np = tensor_to_np(sr_img)
            hr_np = tensor_to_np(hr_img)
            
            # Metrics
            cur_psnr = psnr(hr_np, sr_np, data_range=255)
            cur_ssim = ssim(hr_np, sr_np, data_range=255)
            cur_mse = mse(hr_np, sr_np)

            total_psnr += cur_psnr
            total_ssim += cur_ssim
            total_mse += cur_mse
            
            results_metrics['PSNR'].append(cur_psnr)
            results_metrics['SSIM'].append(cur_ssim)
            results_metrics['MSE'].append(cur_mse)

            count += 1

            # Save visualization for the first few images
            if i < 5:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(tensor_to_np(lr_img), cmap='gray')
                axs[0].set_title('LR Input')
                axs[1].imshow(sr_np, cmap='gray')
                axs[1].set_title(f'SR Output (PSNR: {cur_psnr:.2f})')
                axs[2].imshow(hr_np, cmap='gray')
                axs[2].set_title('HR Target')
                plt.savefig(os.path.join(args.results_dir, f'result_{i}.png'))
                plt.close()

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mse = total_mse / count
    avg_time = total_time / count

    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average Inference Time per image: {avg_time:.6f} seconds")
    
    # Plotting Metric Distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(results_metrics['PSNR'], bins=10, color='blue', alpha=0.7)
    plt.title('PSNR Distribution')
    plt.xlabel('PSNR (dB)')
    
    plt.subplot(1, 3, 2)
    plt.hist(results_metrics['SSIM'], bins=10, color='green', alpha=0.7)
    plt.title('SSIM Distribution')
    plt.xlabel('SSIM')
    
    plt.subplot(1, 3, 3)
    plt.hist(results_metrics['MSE'], bins=10, color='red', alpha=0.7)
    plt.title('MSE Distribution')
    plt.xlabel('MSE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'metrics_distribution.png'))
    print(f"Saved metric graphs to {args.results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT Super Resolution Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing test images')
    parser.add_argument('--model_path', type=str, default='./checkpoints/model_final.pth', help='Path to trained model')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--scale_factor', type=int, default=4, help='Upsampling scale factor')

    args = parser.parse_args()
    evaluate(args)

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from util.SSIM import ssim
from math import log10
from model import HiCENT as hicent
from model.HiCENT_Loss import GeneratorLoss
import utils
import csv

# Training settings
parser = argparse.ArgumentParser(description="HiCENT Training")
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("-nEpochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=20, help="Learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
parser.add_argument("--out_dir", type=str, default="/home/graduates/Betsy/HiCENT/Datasets/checkpoint", help="Output directory")
parser.add_argument("--scale", type=int, default=1, help="Super-resolution scale")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained models")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
args = parser.parse_args()

# Set random seed for reproducibility
torch.backends.cudnn.benchmark = True
seed = args.seed if args.seed is not None else random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if args.cuda else 'cpu')

# Load datasets
def load_data(file_path):
    data = np.load(file_path)
    return torch.tensor(data['data'], dtype=torch.float), torch.tensor(data['target'], dtype=torch.float)

print("===> Loading datasets")
train_data, train_target = load_data('/home/graduates/Betsy/HiCENT/Datasets/data/16hicent_train.npz')
valid_data, valid_target = load_data('/home/graduates/Betsy/HiCENT/Datasets/data/16hicent_valid.npz')

train_set = TensorDataset(train_data, train_target)
valid_set = TensorDataset(valid_data, valid_target)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

# Build model
print("===> Building models")
model = hicent.HICENT(upscale=args.scale)
criterionG = GeneratorLoss().to(device)

if args.cuda:
    model = model.to(device)

# Load pretrained model if exists
def load_pretrained_model(model, pretrained_path):
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"===> Loading pretrained model from '{pretrained_path}'")
        checkpoint = torch.load(pretrained_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if 'module' in k else k
            new_state_dict[name] = v
        model.load_state_dict({k: v for k, v in new_state_dict.items() if k in model.state_dict()}, strict=True)
    else:
        print(f"===> No pretrained model found at '{pretrained_path}'")

load_pretrained_model(model, args.pretrained)

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training function
def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print(f"Epoch = {epoch}, LR = {optimizer.param_groups[0]['lr']}")

    for lr_tensor, hr_tensor in train_loader:
        if args.cuda:
            lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = criterionG(sr_tensor, hr_tensor)
        loss_l1.backward()
        optimizer.step()

    print(f"===> Epoch[{epoch}]: Loss_L1: {loss_l1.item():.5f}")

# Validation function
def valid(best_ssim, best_psnr):
    model.eval()
    valid_result = {'nsamples': 0, 'mse': 0, 'ssims': 0}
    batch_metrics = []

    for lr_tensor, hr_tensor in valid_loader:
        valid_result['nsamples'] += args.batch_size
        if args.cuda:
            lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        batch_mse = ((pre - hr_tensor) ** 2).mean().item()
        batch_psnr = 10 * log10(1 / batch_mse)
        batch_ssim = ssim(pre, hr_tensor).item()

        valid_result['mse'] += batch_mse * args.batch_size
        valid_result['ssims'] += batch_ssim * args.batch_size
        batch_metrics.append({'mse': batch_mse, 'psnr': batch_psnr, 'ssim': batch_ssim})

    valid_result['mse'] /= valid_result['nsamples']
    valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
    valid_result['psnr'] = 10 * log10(1 / valid_result['mse'])

    now_ssim = valid_result['ssim']
    now_psnr = valid_result['psnr']

    # Log metrics to CSV
    csv_file = os.path.join(args.out_dir, 'batch_metrics.csv')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['mse', 'psnr', 'ssim'])
        if file.tell() == 0:  # If the file is new or empty, write the header
            writer.writeheader()
        writer.writerows(batch_metrics)

    if now_ssim > best_ssim or now_psnr > best_psnr:
        if now_ssim > best_ssim:
            best_ssim = now_ssim
            print(f'Best SSIM: {best_ssim:.6f}')
        if now_psnr > best_psnr:
            best_psnr = now_psnr
            print(f'Best PSNR: {best_psnr:.6f}')

        best_ckpt_file = f'psnr{now_psnr:.6f}_ssim{now_ssim:.6f}_best.pytorch'
        torch.save(model.state_dict(), os.path.join(args.out_dir, best_ckpt_file))

    return best_ssim, best_psnr

# Save checkpoint function
def save_checkpoint(epoch):
    model_folder = os.path.join(args.out_dir, f"checkpoint_hicent_x{args.scale}")
    os.makedirs(model_folder, exist_ok=True)
    model_out_path = os.path.join(model_folder, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

# Start training loop
def start_training():
    print("===> Starting training")
    print_network(model)

    best_ssim = 0
    best_psnr = 0

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(epoch)
        best_ssim, best_psnr = valid(best_ssim, best_psnr)
        if epoch % 10 == 0:
            save_checkpoint(epoch)
    finalg_ckpt_file = os.path.join(args.out_dir, 'finalg.pytorch')
    torch.save(model.state_dict(), finalg_ckpt_file)
    print(f"Final model saved as {finalg_ckpt_file}")
# Print network summary
def print_network(net):
    num_params = sum(param.numel() for param in net.parameters())
    print(f'Total number of parameters: {num_params}')

start_training()

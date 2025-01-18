import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import esrt0409_1 as esrt
from util.SSIM import ssim
from math import log10
import utils

# Load model and data for testing
parser = argparse.ArgumentParser(description="ESRT Testing")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--model", type=str, default='ESRT',
                    help="model name")
parser.add_argument("--resume", default="/home/graduates/Betsy/scHiC_data/partition of dataset/chr2s/HFF/H1Esc-HFF.R1/checkpoint/psnr34.446985_ssim0.977920_bestg_10kb40kb_c40_s40_b201_nonpool_hicesrt.pytorch", type=str,
                    help="path to checkpoint")
parser.add_argument("--data_dir", type=str, default="/home/graduates/Betsy/HiCESRT/Data/Valid",
                    help='validation dataset directory')
parser.add_argument("--batch_size", type=int, default=1,
                    help="testing batch size")
parser.add_argument("--scale", type=int, default=1,
                    help="super-resolution scale")
parser.add_argument("--out_dir", type=str, default="/home/graduates/Betsy/scHiC_data/partition of dataset/chr2s/HFF/H1Esc-HFF.R1",
                    help='output directory for results')

args = parser.parse_args()

# Set device
device = torch.device('cuda' if args.cuda else 'cpu')

# Load validation dataset
valid_file = '/home/graduates/Betsy/scHiC_data/partition of dataset/chr2s/HFF/H1Esc-HFF.R1/HFF_test_dataset.npz'
valid_data = np.load(valid_file)

valid_tensor_data = torch.tensor(valid_data['data'], dtype=torch.float)
valid_tensor_target = torch.tensor(valid_data['target'], dtype=torch.float)
valid_inds = torch.tensor(valid_data['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_tensor_data, valid_tensor_target, valid_inds)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

# Load model
model = esrt.ESRT(upscale=args.scale)
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Loading model from '{args.resume}'")
        model.load_state_dict(torch.load(args.resume))
    else:
        print(f"No model found at '{args.resume}'")
model = model.to(device)
model.eval()

# Testing function

def test():
    os.makedirs(args.out_dir, exist_ok=True)  # If the output directory does not exist, create one

    all_sr_outputs = []  # used to save the list of all super-resolution outputs

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor, inds) in enumerate(valid_loader):
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            # Generate super-resolution tensor
            sr_tensor = model(lr_tensor)

            # Convert the generated super-resolution tensor to numpy
            sr_numpy = sr_tensor.cpu().numpy()
            all_sr_outputs.append(sr_numpy)  # Add output to the list

    # concatenate all super-resolution outputs into a numpy array
    all_sr_outputs = np.concatenate(all_sr_outputs, axis=0)

    # Save all super-resolution outputs to one file
    output_file = os.path.join(args.out_dir, "sr_output_test_dataset.npz")
    np.savez_compressed(output_file, data=all_sr_outputs)

    print(f"Save super-resolution outputs to '{output_file}'")

if __name__ == "__main__":
    test()

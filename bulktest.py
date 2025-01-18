import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import HiCENT as hicent
from scipy.stats import pearsonr, spearmanr
import random
from collections import OrderedDict
import datetime
import numpy as np
from util.SSIM import ssim
from math import log10
from util.GenomeDISCO import compute_reproducibility
from util.io import spreadM, together
import time
import multiprocessing
import utils

def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes

# Training settings
parser = argparse.ArgumentParser(description="HiCENT")
parser.add_argument("--batch_size", type=int, default=16, help="training batch size")
parser.add_argument("--best_ssim", type=int, default=0)
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda")
parser.add_argument("--cell_line", type=str, default="GM12878_HiCENT", help='dataset directory')
parser.add_argument("--out_dir", type=str, default="/home/graduates/Betsy/HiCENT/Datasets/predict", help='dataset directory')
parser.add_argument("--scale", type=int, default=1, help="super-resolution scale")
parser.add_argument("--lowres", dest='low_res', type=str, default='40kb', help="output patch size")
parser.add_argument("--pretrained", default="/home/graduates/Betsy/HiCENT/Datasets/checkpoint/psnr35.776812_ssim0.918952_best.pytorch", type=str, help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--model", type=str, default='HICENT')

args = parser.parse_args()

args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

# prepare valid dataset
valid_file = '/home/graduates/Betsy/HiCENT/Datasets/data/16hicent_GM12878_test.npz'
valid = np.load(valid_file, allow_pickle=True)

indices, compacts, sizes = data_info(valid)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched validation
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

print("===> Building models")
args.is_train = True

model = hicent.HICENT(upscale=args.scale)

l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=True)
    else:
        print("===> no models found at '{}'".format(args.pretrained))

def forward_chop(model, x, scale, shave=10, min_size=60000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]
    ]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def valid(scale, best_ssim):
    model.eval()
    result_data = []
    result_inds = []
    chr_nums = sorted(list(np.unique(valid_inds[:, 0])))

    results_dict = dict()
    test_metrics = dict()
    for chr in chr_nums:
        test_metrics[f'{chr}'] = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0, 'pccs': 0, 'pcc': 0,
                                  'spcs': 0, 'spc': 0, 'snrs': 0, 'snr': 0}
        results_dict[f'{chr}'] = [[], [], [], [], [], [], []]

    for lr_tensor, hr_tensor, inds in valid_loader:
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = forward_chop(model, lr_tensor, scale)

        ind = f'{(inds[0][0]).item()}'
        batch_size = lr_tensor.size(0)
        test_metrics[ind]['nsamples'] += batch_size

        batch_mse = ((pre - hr_tensor) ** 2).mean()
        test_metrics[ind]['mse'] += batch_mse * batch_size
        batch_ssim = ssim(pre, hr_tensor)
        test_metrics[ind]['ssims'] += batch_ssim * batch_size
        test_metrics[ind]['psnr'] = 10 * log10(1 / (test_metrics[ind]['mse'] / test_metrics[ind]['nsamples']))
        test_metrics[ind]['ssim'] = test_metrics[ind]['ssims'] / test_metrics[ind]['nsamples']
        batch_snr = (hr_tensor.sum() / ((hr_tensor - pre) ** 2).sum().sqrt())
        test_metrics[ind]['snrs'] += batch_snr * batch_size
        test_metrics[ind]['snr'] = test_metrics[ind]['snrs']
        batch_pcc = pearsonr(pre.cpu().flatten(), hr_tensor.cpu().flatten())[0]
        batch_spc = spearmanr(pre.cpu().flatten(), hr_tensor.cpu().flatten())[0]
        test_metrics[ind]['pccs'] += batch_pcc * batch_size
        test_metrics[ind]['spcs'] += batch_spc * batch_size
        test_metrics[ind]['pcc'] = test_metrics[ind]['pccs'] / test_metrics[ind]['nsamples']
        test_metrics[ind]['spc'] = test_metrics[ind]['spcs'] / test_metrics[ind]['nsamples']

        ((results_dict[ind])[0]).append((test_metrics[ind]['ssim']).item())
        ((results_dict[ind])[1]).append(batch_mse.item())
        ((results_dict[ind])[2]).append(test_metrics[ind]['psnr'])
        ((results_dict[ind])[4]).append(test_metrics[ind]['snr'].item())
        ((results_dict[ind])[5]).append(test_metrics[ind]['pcc'])
        ((results_dict[ind])[6]).append(test_metrics[ind]['spc'])

        for i, j in zip(hr_tensor, pre):
            out1 = torch.squeeze(j, dim=0)
            hr1 = torch.squeeze(i, dim=0)
            out2 = out1.cpu().detach().numpy()
            hr2 = hr1.cpu().detach().numpy()
            genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
            ((results_dict[ind])[3]).append(genomeDISCO)

        result_data.append(pre.to('cpu').numpy())
        result_inds.append(inds.numpy())

    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)

    mean_ssims, mean_mses, mean_psnrs, mean_gds, mean_snr, mean_pcc, mean_spc = [], [], [], [], [], [], []

    for key, value in results_dict.items():
        value[0] = round(sum(value[0]) / len(value[0]), 4)
        value[1] = round(sum(value[1]) / len(value[1]), 4)
        value[2] = round(sum(value[2]) / len(value[2]), 4)
        value[3] = round(sum(value[3]) / len(value[3]), 4)
        value[4] = round(sum(value[4]) / len(value[4]), 4)
        value[5] = round(sum(value[5]) / len(value[5]), 4)
        value[6] = round(sum(value[6]) / len(value[6]), 4)

        mean_ssims.append(value[0])
        mean_mses.append(value[1])
        mean_psnrs.append(value[2])
        mean_gds.append(value[3])
        mean_snr.append(value[4])
        mean_pcc.append(value[5])
        mean_spc.append(value[6])

        print("\nChr", key, "SSIM: ", value[0])
        print("Chr", key, "MSE: ", value[1])
        print("Chr", key, "PSNR: ", value[2])
        print("Chr", key, "GenomeDISCO: ", value[3])
        print("Chr", key, "SNR: ", value[4])
        print("Chr", key, "PCC: ", value[5])
        print("Chr", key, "SpC: ", value[6])

    print("\n___________________________________________")
    print("Means across chromosomes")
    print("SSIM: ", round(sum(mean_ssims) / len(mean_ssims), 4))
    print("MSE: ", round(sum(mean_mses) / len(mean_mses), 4))
    print("PSNR: ", round(sum(mean_psnrs) / len(mean_psnrs), 4))
    print("GenomeDISCO: ", round(sum(mean_gds) / len(mean_gds), 4))
    print("SNR: ", round(sum(mean_snr) / len(mean_snr), 4))
    print("PCC: ", round(sum(mean_pcc) / len(mean_pcc), 4))
    print("SPC: ", round(sum(mean_spc) / len(mean_spc), 4))
    print("___________________________________________\n")

    hicent_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return hicent_hics

def save_data(carn, compact, size, file):
    hicent = spreadM(carn, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hicent=hicent, compact=compact)
    print('Saving file:', file)

start = time.time()
print("===> Testing")
code_start = datetime.datetime.now()
timer = utils.Timer()
best_ssim = 0

if multiprocessing.cpu_count() > 23:
    pool_num = 23
else:
    exit()

t_epoch_start = timer.t()

hicesrt_hics = valid(args.scale, best_ssim)

def save_data_n(key):
    file = os.path.join(args.out_dir, args.cell_line, f'predict_chr{key}_{args.low_res}.npz')
    save_data(hicesrt_hics[key], compacts[key], sizes[key], file)

pool = multiprocessing.Pool(processes=pool_num)
print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
for key in compacts.keys():
    pool.apply_async(save_data_n, (key,))
pool.close()
pool.join()
print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')

import os
import argparse

# the Root directory for all raw and processed data
root_dir = '/home/graduates/Betsy/HiCENT/Datasets'  # Example of root directory name

res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000,
           '500kb': 500_000, '1mb': 1_000_000}

# 'train' and 'valid' can be changed for different train/valid set splitting
set_dict = {'K562_test': [3, 11, 19, 21],
            'mESC_test': (4, 9, 15, 18),
            'train': [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22],
            'valid': [2, 6, 10, 12],
            'GM12878_test': (4, 14, 16, 20)}

help_opt = (('--help', '-h'), {
    'action': 'help',
    'help': "Print this help message and exit"})


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


# chr12_10kb.npz, predict_chr13_40kb.npz
def chr_num_str(x):
    start = x.find('chr')
    part = x[start + 3:]
    end = part.find('_')
    return part[:end]


def chr_digit(filename):
    chrn = chr_num_str(os.path.basename(filename))
    if chrn == 'X':
        n = 23
    else:
        n = int(chrn)
    return n


def data_read_parser():
    parser = argparse.ArgumentParser(description='Read raw data from Rao\'s Hi-C.', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          default='GM12878')

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]',
                           default='10kb', choices=res_map.keys())
    misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',
                           default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
    misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]',
                           default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_down_parser():
    parser = argparse.ArgumentParser(description='Downsample data from high resolution data', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          default='GM12878')
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
                          default='10kb', choices=res_map.keys())
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
                          default='40kb')
    req_args.add_argument('-r', dest='ratio', help='REQUIRED: The ratio of downsampling[example:16]',
                          default=16, type=int)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_divider_parser():
    parser = argparse.ArgumentParser(description='Divide data for train and predict', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
                          default='GM12878')
    req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
                          default='10kb', choices=res_map.keys())
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
                          default='40kb')
    req_args.add_argument('-lrc', dest='lr_cutoff', help='REQUIRED: cutoff for low resolution maps[example:100]',
                          default=100, type=int)
    req_args.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)',
                          default='train', choices=['K562_test', 'mESC_test', 'train', 'valid', 'GM12878_test'], )
    hicent_args = parser.add_argument_group('HiCENT Arguments')
    hicent_args.add_argument('-chunk', dest='chunk', help='REQUIRED: chunk size for dividing[example:40]',
                              default=40, type=int)
    hicent_args.add_argument('-stride', dest='stride', help='REQUIRED: stride for dividing[example:40]',
                              default=40, type=int)
    hicent_args.add_argument('-bound', dest='bound', help='REQUIRED: distance boundary interested[example:201]',
                              default=201, type=int)
    hicent_args.add_argument('-scale', dest='scale', help='REQUIRED: Downpooling scale[example:1]',
                              default=1, type=int)
    hicent_args.add_argument('-type', dest='pool_type', help='OPTIONAL: Downpooling type[default:max]',
                              default='max', choices=['max', 'avg'])
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser


def data_predict_parser():
    parser = argparse.ArgumentParser(description='Predict data using HiCARN model', add_help=False)
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example: GM12878]',
                          required=True)
    req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example: 40kb]',
                          default='40kb', required=True)
    req_args.add_argument('-f', dest='file_name', help='REQUIRED: Matrix file to be enhanced[example: '
                                                       'hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz', required=True)
    req_args.add_argument('-m', dest='model', help='REQUIRED: Choose your model[example: HiCARN_1]', required=True)
    gan_args = parser.add_argument_group('GAN model Arguments')
    gan_args.add_argument('-ckpt', dest='checkpoint', help='REQUIRED: Checkpoint file of HiCARN model',
                          required=True)
    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('--cuda', dest='cuda', help='Whether or not using CUDA[default:1]',
                           default=1, type=int)
    parser.add_argument(*help_opt[0], **help_opt[1])

    return parser

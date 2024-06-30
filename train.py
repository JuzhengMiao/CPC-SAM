# training code
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from segment_anything import sam_model_registry

from trainer_dualmask import trainer_acdc_dualmask_prompt_ssl_fixcoe_random_new_mean_up
from icecream import ic
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='xxx/data/ACDC', help='root dir for data')
parser.add_argument('--output', type=str, default='xxx/output/sam/results_ssl')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--num_classes', type=int,   # the number of foreground classes
                    default=3, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')  # can change to a larger number for multi-gpu training
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1337, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b_dualmask_same_prompt_class_random_large', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='xxx/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr', default=True)
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model', default=True)
parser.add_argument('--module', type=str, default='sam_lora_image_encoder_prompt')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--labeled_num', type=int, default=1,help='labeled data')  # determine the number of labeled data
parser.add_argument('--warm_iter', type=int, default=5000,help='labeled data') 
parser.add_argument('--method', type=str, default='ssl_dualmask_same_warm5000_pointprompt_new_fixcoe_class_random_new_c2_mean_large_A40_20240219')
parser.add_argument('--promptmode', type=str, default='point',help='prompt')

parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=600.0, help='consistency_rampup')
parser.add_argument('--T', type=float,
                    default=0.1, help='temperature')
parser.add_argument('--coe', type=float,
                    default=0.4, help='coe')
parser.add_argument('--coe2', type=float,
                    default=0.05, help='coe')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'ACDC': {
            'root_path': args.root_path,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.005 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_dice_coeff' + str(args.dice_param) if args.dice_param != 0.8 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.labeled_num)+'_labeled'
    snapshot_path += '_' + args.method
    snapshot_path += '_T' + str(args.T)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    multimask_output = True

    low_res = img_embedding_size * 4   

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'ACDC': trainer_acdc_dualmask_prompt_ssl_fixcoe_random_new_mean_up}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)

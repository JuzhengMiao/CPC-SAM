# ensemble
import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume_mean
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_ACDC import ACDC_dataset
import h5py
from icecream import ic
import pandas as pd

def inference(args, multimask_output, db_config, model, test_save_path=None):
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    metric_list = 0.0
     
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    first_list = np.zeros([len(image_list), 4])
    second_list = np.zeros([len(image_list), 4])
    third_list = np.zeros([len(image_list), 4])
    count = 0

    for case in tqdm(image_list):
        h5f = h5py.File(args.root_path + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        metric_i = test_single_volume_mean(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case= case,z_spacing=db_config['z_spacing'])
        first_metric = metric_i[0]
        second_metric = metric_i[1]
        third_metric = metric_i[2]

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)

        # save
        first_list[count, 0]=first_metric[0]
        first_list[count, 1]=first_metric[1]
        first_list[count, 2]=first_metric[2]
        first_list[count, 3]=first_metric[3]

        second_list[count, 0]=second_metric[0]
        second_list[count, 1]=second_metric[1]
        second_list[count, 2]=second_metric[2]
        second_list[count, 3]=second_metric[3]

        third_list[count, 0]=third_metric[0]
        third_list[count, 1]=third_metric[1]
        third_list[count, 2]=third_metric[2]
        third_list[count, 3]=third_metric[3]

        count += 1

    avg_metric1 = np.nanmean(first_list, axis=0)
    avg_metric2 = np.nanmean(second_list, axis=0)
    avg_metric3 = np.nanmean(third_list, axis=0)
    
    # save:
    write_csv = "xxx/output/save_excel/" + args.exp + "_test_mean.csv"
    save = pd.DataFrame({'RV-dice':first_list[:,0], 'RV-hd95':first_list[:,1], 'RV-asd':first_list[:,2], 'RV-jc':first_list[:,3], 'Myo-dice':second_list[:,0], 'Myo-hd95':second_list[:,1], 'Myo-asd':second_list[:,2], 
    'Myo-jc':second_list[:,3], 'LV-dice':third_list[:,0], 'LV-hd95':third_list[:,1], 'LV-asd':third_list[:,2], 'LV-jc':third_list[:,3]})
    save.to_csv(write_csv, index=False, sep=',')  
    
    print(avg_metric1, avg_metric2, avg_metric3)
    print((avg_metric1+avg_metric2+avg_metric3)/3)
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                    default='xxx/data/ACDC', help='Name of Experiment')
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--dataset', type=str, default='ACDC', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='xxx/')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1337, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', default=False,help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='xxx/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    
    parser.add_argument('--lora_ckpt', type=str, default='xxx/checkpoint/best_model.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b_dualmask_same_prompt_class_random_large', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder_prompt')
    parser.add_argument('--exp', type=str, default='test_name')
    parser.add_argument('--promptmode', type=str, default='point',help='prompt')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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
            'Dataset': ACDC_dataset,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path=test_save_path)


import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, sigmoid_rampup
from torchvision import transforms
from icecream import ic
from val import test_single_volume
from datasets.dataset_ACDC import TwoStreamBatchSampler
import ipdb
from PIL import Image
from skimage.measure import label

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    # for the first branch
    low_res_logits1 = outputs['low_res_logits1']
    loss_ce1 = ce_loss(low_res_logits1, low_res_label_batch[:].long())
    loss_dice1 = dice_loss(low_res_logits1, low_res_label_batch, softmax=True)
    loss1 = (1 - dice_weight) * loss_ce1+ dice_weight * loss_dice1

    # for the second branch
    low_res_logits2 = outputs['low_res_logits2']
    loss_ce2 = ce_loss(low_res_logits2, low_res_label_batch[:].long())
    loss_dice2 = dice_loss(low_res_logits2, low_res_label_batch, softmax=True)
    loss2 = (1 - dice_weight) * loss_ce2+ dice_weight * loss_dice2

    loss = loss1 + loss2
    return loss, loss1, loss_ce1, loss_dice1, loss2, loss_ce2, loss_dice2

def calc_loss_labeled(low_res_logits, low_res_label_batch, ce_loss, dice_loss, labeled_bs,dice_weight:float=0.8):
    low_res_logits_labeled = low_res_logits[:labeled_bs]
    low_res_label_batch_labeled = low_res_label_batch[:labeled_bs]

    loss_ce = ce_loss(low_res_logits_labeled, low_res_label_batch_labeled[:].long())
    loss_dice = dice_loss(low_res_logits_labeled, low_res_label_batch_labeled, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def get_current_consistency_weight(epoch,consistency,max_iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency_rampup = int( max_iter * 200.0 / 30000.0 )
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 200 # int(max_epoch/6)          
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def trainer_acdc_dualmask_prompt_ssl_fixcoe_random_new_mean_up(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    labeled_bs = batch_size//2 #args.labeled_bs * args.n_gpu
    print('labeled_bs: ', labeled_bs)
    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    # max_iterations = args.max_iterations
    db_train = ACDC_dataset(base_dir=args.root_path,split="train",num=None,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    
    db_val = ACDC_dataset(base_dir=args.root_path, split="val")
    
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  #  [b, c, h, w], [b, h, w]
            
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'

            # the first round
            
            outputs = model(image_batch, multimask_output, args.img_size, -1, args.promptmode)

            outputs1 = outputs['low_res_logits1']
            outputs2 = outputs['low_res_logits2']

            supervised_loss1, loss_ce1, loss_dice1 = calc_loss_labeled(outputs1, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)
            supervised_loss2, loss_ce2, loss_dice2 = calc_loss_labeled(outputs2, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)
            loss1 = supervised_loss1 + supervised_loss2
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            consistency_weight = args.coe 
            T = args.T

            # the second round
            if iter_num < args.warm_iter:
                supervised_round2_loss1, loss_round2_ce1, loss_round2_dice1 = 0.0, 0.0, 0.0
                supervised_round2_loss1_r, loss_round2_ce1_r, loss_round2_dice1_r = 0.0, 0.0, 0.0
            else:    
                outputs_round2 = model(image_batch, multimask_output, args.img_size, 1, args.promptmode)
                outputs_round2_1 = outputs_round2['low_res_logits1']
                outputs_round2_1_r = outputs_round2['low_res_logits1_r']
                outputs_round2_2 = outputs_round2['low_res_logits2']
                outputs_round2_soft1 = torch.softmax(outputs_round2_1, dim=1)
                outputs_round2_soft1_r = torch.softmax(outputs_round2_1_r, dim=1)
                outputs_round2_soft2 = torch.softmax(outputs_round2_2, dim=1)

                supervised_round2_loss1, loss_round2_ce1, loss_round2_dice1 = calc_loss_labeled(outputs_round2_1, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)

                supervised_round2_loss1_r, loss_round2_ce1_r, loss_round2_dice1_r = calc_loss_labeled(outputs_round2_1_r, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)

            if iter_num < args.warm_iter:
                consistency_loss2 = 0.0
                consistency_loss1_r = 0.0
                loss2 = 0.0
            else:
                outputs_round2_soft1 = (outputs_round2_soft1 + outputs_round2_soft1_r)/2.0
                pseudo_outputs1 = torch.argmax(outputs_round2_soft1[labeled_bs:].detach(), dim=1, keepdim=False)
               
                consistency_loss2 = 0.5*(ce_loss(outputs_round2_2[labeled_bs:], pseudo_outputs1.long()) + dice_loss(outputs_round2_2[labeled_bs:], pseudo_outputs1, softmax=True))

                consistency_loss1_r = 0.5*(ce_loss(outputs_round2_1_r[labeled_bs:], pseudo_outputs1.long()) + dice_loss(outputs_round2_1_r[labeled_bs:], pseudo_outputs1, softmax=True))

                loss2 = supervised_round2_loss1 + supervised_round2_loss1_r + consistency_weight * consistency_loss2 + args.coe2 * consistency_loss1_r
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

            # the third round
            if iter_num < args.warm_iter:
                supervised_round3_loss1, loss_round3_ce1, loss_round3_dice1 = 0.0, 0.0, 0.0
                supervised_round3_loss1_r, loss_round3_ce1_r, loss_round3_dice1_r = 0.0, 0.0, 0.0

            else:
                outputs_round3 = model(image_batch, multimask_output, args.img_size, 0, args.promptmode) 
                outputs_round3_1 = outputs_round3['low_res_logits1']
                outputs_round3_2 = outputs_round3['low_res_logits2']
                outputs_round3_2_r = outputs_round3['low_res_logits2_r']
                outputs_round3_soft1 = torch.softmax(outputs_round3_1, dim=1)
                outputs_round3_soft2 = torch.softmax(outputs_round3_2, dim=1)
                outputs_round3_soft2_r = torch.softmax(outputs_round3_2_r, dim=1)

                supervised_round3_loss1, loss_round3_ce1, loss_round3_dice1 = calc_loss_labeled(outputs_round3_2, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)
                supervised_round3_loss1_r, loss_round3_ce1_r, loss_round3_dice1_r = calc_loss_labeled(outputs_round3_2_r, label_batch, ce_loss, dice_loss, labeled_bs,args.dice_param)

            if iter_num < args.warm_iter:
                consistency_loss1 = 0.0
                consistency_loss2_r = 0.0
                loss3 = 0.0
            else:
                outputs_round3_soft2 = (outputs_round3_soft2 + outputs_round3_soft2_r)/2.0
                pseudo_outputs2 = torch.argmax(outputs_round3_soft2[labeled_bs:].detach(), dim=1, keepdim=False)
                
                consistency_loss1 = 0.5*(ce_loss(outputs_round3_1[labeled_bs:], pseudo_outputs2.long()) + dice_loss(outputs_round3_1[labeled_bs:], pseudo_outputs2, softmax=True))
                consistency_loss2_r = 0.5*(ce_loss(outputs_round3_2_r[labeled_bs:], pseudo_outputs2.long()) + dice_loss(outputs_round3_2_r[labeled_bs:], pseudo_outputs2, softmax=True))

                loss3 = supervised_round3_loss1 + supervised_round3_loss1_r + consistency_weight * consistency_loss1 + args.coe2 * consistency_loss2_r
                optimizer.zero_grad()
                loss3.backward()
                optimizer.step()

            loss = loss1 + loss2 + loss3

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # the first round
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss1', loss1, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)

            # the second round
            writer.add_scalar('info/loss2', loss2, iter_num)
            # loss_round2_ce1, loss_round2_dice1
            writer.add_scalar('info/loss_round2_ce1', loss_round2_ce1, iter_num)
            writer.add_scalar('info/loss_round2_dice1', loss_round2_dice1, iter_num)
            writer.add_scalar('info/loss_round2_ce1_r', loss_round2_ce1_r, iter_num)
            writer.add_scalar('info/loss_round2_dice1_r', loss_round2_dice1_r, iter_num)
            writer.add_scalar('info/consistency_loss2',
                              consistency_loss2, iter_num)
            writer.add_scalar('info/consistency_loss1_r',
                              consistency_loss1_r, iter_num)
            
            # the third round
            writer.add_scalar('info/loss3', loss3, iter_num)
            # loss_round2_ce1, loss_round2_dice1
            writer.add_scalar('info/loss_round3_ce1', loss_round3_ce1, iter_num)
            writer.add_scalar('info/loss_round3_dice1', loss_round3_dice1, iter_num)
            writer.add_scalar('info/loss_round3_ce1_r', loss_round3_ce1_r, iter_num)
            writer.add_scalar('info/loss_round3_dice1_r', loss_round3_dice1_r, iter_num)
            writer.add_scalar('info/consistency_loss1',
                              consistency_loss1, iter_num)
            writer.add_scalar('info/consistency_loss2_r',
                              consistency_loss2_r, iter_num)

            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            if iter_num < args.warm_iter + 1:
                logging.info('iteration %d : loss : %f, loss1: %f' % (iter_num, loss.item(), loss1.item()))
            else:
                logging.info('iteration %d : loss : %f, loss1: %f, loss2: %f, loss3: %f' % (iter_num, loss.item(), loss1.item(), loss2.item(), loss3.item()))

            if iter_num > 0 and iter_num % 200 == 0:  # evaluation
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes+1) 
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes):   
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                    metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                    metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                            'best_model.pth')
                    try:
                        model.save_lora_parameters(save_best)
                        model.save_lora_parameters(save_mode_path)
                    except:
                        model.module.save_lora_parameters(save_best)
                        model.module.save_lora_parameters(save_mode_path)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
    

        save_interval = 200 # int(max_epoch/6)   # 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

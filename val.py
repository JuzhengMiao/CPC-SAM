import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from einops import repeat


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, multimask_output=True, patch_size=[512, 512]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(input, 'b c h w -> b (repeat c) h w', repeat=3)
        
        net.eval()

        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_image(image, label, net, classes, multimask_output=True, patch_size=[512, 512]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    # for ind in range(image.shape[0]):

    slice = image
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(slice).unsqueeze(
        0).unsqueeze(0).float().cuda()
    inputs = repeat(input, 'b c h w -> b (repeat c) h w', repeat=3)
    
    net.eval()

    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        out_h, out_w = out.shape
        if x != out_h or y != out_w:
            pred = zoom(out, (x / out_h, y / out_w), order=0)
        else:
            pred = out
        prediction = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_prompt(image, label, net, classes, promptidx, promptmode, multimask_output=True, patch_size=[512, 512]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(input, 'b c h w -> b (repeat c) h w', repeat=3)
        
        net.eval()

        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0], promptidx, promptmode)
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list



def test_single_volume_scm(image, label, net1,net2,scm, classes, multimask_output=True, patch_size=[512, 512]):
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(input, 'b c h w -> b (repeat c) h w', repeat=3)
        
        net1.eval()
        net2.eval()
        scm.eval()

        with torch.no_grad():

            outputs1 = net1(inputs, multimask_output, patch_size[0])
            output_masks1 = outputs1['masks']
            output1 = torch.softmax(output_masks1,dim=1)

            outputs2 = net2(inputs, multimask_output, patch_size[0])
            output_masks2 = outputs2['masks']
            output2 = torch.softmax(output_masks2,dim=1)

            conv_in = torch.cat([output1, output2], dim = 1)
            conv_out = scm(conv_in)

            out = torch.argmax(torch.softmax(conv_out, dim=1), dim=1).squeeze(0)
            
            out  = out .cpu().detach().numpy()

            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
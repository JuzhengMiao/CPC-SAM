# each class has its corresponding point embedding
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_prompt_large import MaskDecoder_prompt_large
from .prompt_encoder_prompt_class import PromptEncoder_prompt_class
import numpy as np

from skimage.measure import label
import cv2

def MaskToBoxSimple(mask):
    mask = mask.squeeze()
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()

    return [x0,y0,x1,y1]

class Sam_dualmask_same_prompt_class_random_large(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder_prompt_class,
        mask_decoder1: MaskDecoder_prompt_large,
        mask_decoder2: MaskDecoder_prompt_large,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder1 = mask_decoder1
        self.mask_decoder2 = mask_decoder2
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size, prompt_idx = -1, prompt_mode = None):
        # prompt_idx indicates which branch is used to generate prompts
        if isinstance(batched_input, list):
            outputs = self.forward_test(batched_input, multimask_output)
        else:
            outputs = self.forward_train(batched_input, multimask_output, image_size, prompt_idx, prompt_mode)  
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size, prompt_idx, prompt):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)

        if prompt_idx == 0:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            # generate prompts based on the coarse prediction
            points_prompt, points_prompt_random, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks1, image_size, True)

            if prompt == 'point':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=None
                )
                sparse_embeddings_r, _ = self.prompt_encoder(
                    points=points_prompt_random, boxes=None, masks=None
                )
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )


            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks2_r, iou_predictions2_r, _ = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_r,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )
        
        elif prompt_idx == 1:  
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
        
            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output 
            )

            # generate prompts based on the coarse prediction
            points_prompt, points_prompt_random, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks2, image_size, True) 

            if prompt == 'point':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=None
                )
                sparse_embeddings_r, dense_embeddings_r = self.prompt_encoder(
                    points=points_prompt_random, boxes=None, masks=None
                )
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks1_r, iou_predictions1_r, _ = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_r,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )
        
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
            )

            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )


        masks1 = self.postprocess_masks(
            low_res_masks1,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        masks2 = self.postprocess_masks(
            low_res_masks2,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        if prompt_idx != -1:
            if prompt_idx == 1:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'low_res_logits1_r': low_res_masks1_r,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2
                }
            else:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2,
                    'low_res_logits2_r': low_res_masks2_r
                }
        else:
            outputs = {
                'masks': masks1,
                'iou_predictions1': iou_predictions1,
                'low_res_logits1': low_res_masks1,
                'masks2': masks2,
                'iou_predictions2': iou_predictions2,
                'low_res_logits2': low_res_masks2
            }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions, _ = self.mask_decoder1(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def prompt_generate_random_fast(self, coarse_mask, img_size, israndom = False):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape

        coarse_mask_np = torch.argmax(coarse_mask, dim = 1)
        coarse_mask_np = F.interpolate(coarse_mask_np.unsqueeze(1).float(), (img_size, img_size), mode="nearest").squeeze(1)
        coarse_mask_np = coarse_mask_np.detach().cpu().numpy()

        # points: BxNx2 tensor & boxes
        points_prompt = np.zeros([b, num_class, 2])
        points_label = np.zeros([b, num_class])
        points_prompt_random = np.zeros([b, num_class, 2])
        for idx in range(b):  # iterate over each image
            for cls in range(num_class): # find points for each class
                # obtain the binary mask
                mask_cls = (coarse_mask_np[idx] == cls).astype(np.uint8)
                if mask_cls.max() > 0:
                    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
                    ratio_list, regionid_list = [], []
                    for region_id in range(1, region_ids+1):
                        #find coordinates of points in the region
                        binary_msk = np.where(label_msk==region_id, 1, 0)

                        # clean some region that is abnormally small
                        r = np.sum(binary_msk) / np.sum(mask_cls)
                        # print('curr mask over all mask ratio', r)
                        ratio_list.append(r)
                        regionid_list.append(region_id)

                    ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
                    regionid_list = regionid_list[::-1]

                    binary_msk = np.where(label_msk==regionid_list[0], 1, 0)
                    
                    if israndom:  
                        cY_r, cX_r = np.where(binary_msk==1)
                        random_idx = np.random.randint(0, len(cX_r))
                        points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = int(cX_r[random_idx]), int(cY_r[random_idx])

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
                    dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                    cY, cX = np.where(dist_img==dist_img.max())
                    random_idx = np.random.randint(0, len(cX))
                    points_prompt[idx,cls,0], points_prompt[idx,cls,1] = int(cX[random_idx]), int(cY[random_idx])
                    
                    if cls > 0:
                        points_label[idx,cls] = cls
                
                else:
                    points_prompt[idx,cls,0], points_prompt[idx,cls,1] = points_prompt[idx,0,0], points_prompt[idx,0,1]
                    points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = points_prompt[idx,0,0], points_prompt[idx,0,1]
                    points_label[idx,cls] = 0

        points_prompt = torch.tensor(points_prompt).to(coarse_mask.device)
        points_label = torch.tensor(points_label).to(coarse_mask.device)
        points_prompt = (points_prompt, points_label)

        if israndom:  
            points_prompt_random = torch.tensor(points_prompt_random).to(coarse_mask.device)
            points_prompt_random = (points_prompt_random, points_label)

            return points_prompt, points_prompt_random, None, None

        return points_prompt, None, None

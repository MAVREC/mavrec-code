# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# https://github.com/ahmed-nady/Deformable-DETR
#helped by chatgpt in visualisation

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T
from pycocotools.coco import COCO

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models import build_model_semi
import time
import os


import math

from PIL import Image
import requests
import matplotlib.pyplot as plt


import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

def get_args_parser():
    parser = argparse.ArgumentParser('Omni-DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=150, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes including background')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=16, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=900, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file',
                        default='coco_omni')  # coco_omni, coco_35to80_tagsU, coco_35to80_point, coco_objects_tagsU, coco_objects_points, bees_omni, voc_semi_ voc_omni, objects_omni, crowdhuman_omni
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='results_tmp',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--percent', default='10',
                        help='percent with fully labeled')
    parser.add_argument('--BURN_IN_STEP', default=20, type=int,
                        help='as the name means')
    parser.add_argument('--TEACHER_UPDATE_ITER', default=1, type=int,
                        help='as the name means')
    parser.add_argument('--EMA_KEEP_RATE', default=0.9996, type=float,
                        help='as the name means')
    # labelled
    parser.add_argument('--label_label_root', default='supervised_annotations/aerial/aligned_ids/',
                        help='percent with fully labeled')
    parser.add_argument('--unlabel_label_root', default='',
                        help='percent with fully labeled')
    parser.add_argument('--annotation_json_train_label', default='',
                        help='percent with fully labeled')
    parser.add_argument('--annotation_json_train_unlabel', default='',
                        help='percent with fully labeled')
    parser.add_argument('--annotation_json_val', default='',
                        help='percent with fully labeled')
    # data baths
    parser.add_argument('--data_dir_label_train', default='',
                        help='percent with fully labeled')
    parser.add_argument('--data_dir_unlabel_train', default='',
                        help='percent with fully labeled')
    parser.add_argument('--data_dir_label_val', default='',
                        help='percent with fully labeled')
    
    parser.add_argument('--CONFIDENCE_THRESHOLD', default=0.7, type=float,
                        help='as the name means')
    parser.add_argument('--PASSED_NUM', default='passed_num.txt',
                        help='the sample number passing the threshold each batch')
    parser.add_argument('--pixels', default=600, type=int,
                        help='as the name means')
    parser.add_argument('--w_p', default=0.5, type=float,
                        help='weight for point cost matrix during filtering')
    parser.add_argument('--w_t', default=0.5, type=float,
                        help='weight for tag cost matrix during filtering')
    parser.add_argument('--save_freq', default=5, type=int,
                        metavar='N', help='checkpoint save frequency (default: 5)')
    parser.add_argument('--eval_freq', default=1, type=int)
    return parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    #fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion, postprocessors=build_model(args)
    model.to(device)
    RESUME_PATH = '/home/rchakra6/OMNIDETR_PROJECT/dvdv1-code/omni-detr/results/vanilla_omni_detr_rajat/checkpoint0039.pth'
    checkpoint = torch.load(RESUME_PATH, map_location='cpu')
    print(checkpoint.keys())

    model.load_state_dict(checkpoint['model_teacher'], strict=False)
    if torch.cuda.is_available():
         model.cuda()
    model.eval()
    

    

    t0 = time.time()
    DATA_DIR = '/data/SDU-GAMODv4/SDU-GAMODv4/scaled_dataset/train/droneview/'

    #read file
    dataDir='/data/SDU-GAMODv4/SDU-GAMODv4/supervised_annotations/aerial/aligned_ids/'
    dataType = 'aerial_train_aligned_ids_w_indicator_with_perspective_with_points' #cocoFIle
    annFile='{}{}.json'.format(dataDir,dataType)

    # Initialize the COCO api for instance annotations
    coco=COCO(annFile)

    
    for img_id in coco.getImgIds():
        image = coco.loadImgs(img_id)[0]
        print("Processing Image: ", image['file_name'])
        im = Image.open(DATA_DIR + image['file_name'])
        img = transform(im).unsqueeze(0)

        img=img.cuda()
        # propagate through the model
        outputs =  model(img)

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        #print("out_logits", out_logits)
        #print("out_bbox:", out_bbox)

        prob = out_logits.sigmoid()
        #breakpoint()
        
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        keep = scores[0] > 0.70 # this is the thresholds for showing this
        boxes = boxes[0, keep]
        labels = labels[0, keep]
        scores = scores[0, keep] # probabilities

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h, im_w = im.size
        target_sizes =torch.tensor([[im_w,im_h]])
        target_sizes =target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        print(time.time()-t0)

        # Load the categories in a variable
        catIDs = coco.getCatIds()
        categories = coco.loadCats(catIDs)

        # plot_results
        source_img = Image.open(DATA_DIR+IMG['file_name']).convert("RGBA")
        draw = ImageDraw.Draw(source_img)

        ANNO_COLOR = (255, 0, 0) #red for prediction
        # DRAW PREDICTIONS
        index = 0
        for xmin, ymin, xmax, ymax in boxes[0].tolist():
            category_label_id =  labels[index].item()
            # find the corresponding category name:
            if category_label_id != 0:
                category = coco.loadCats(category_label_id)
                # print("CAT", category)
                category_label_name = str(category[0]['name'])

                probability_score = format(scores[index].item()*100, ".2f") # softmax
                print_text = str(category_label_name) #+ ',' + str(probability_score)
                #print( str(category_label_name), " with prob.: ", probability_score)
                # draw text
                draw.text((xmin, ymin), print_text, fill= ANNO_COLOR)
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline= ANNO_COLOR)
                index += 1

        # DRAW GROUND TRUTH
        anno_ids = coco.getAnnIds(imgIds=[IMG['id']])
        print("anno_ids: ", anno_ids)
        gt_annos = coco.loadAnns(ids=anno_ids)

        gt_boxes = []
        for gt_anno in gt_annos:
            bbox = gt_anno['bbox']
            # print("bbox: ", bbox)
            gt_boxes.append(bbox)

        GT_ANNO_COLOR = (0, 255, 0)#green for groundtruth
        for xmin, ymin, xmax, ymax in gt_boxes:
            
                draw.rectangle(((xmin, ymin), (xmin+xmax, ymin+ymax)), outline = GT_ANNO_COLOR)
                # index += 1
        
        filename = 'teacher_400.txt'
        #Create or open the file in append mode
        with open(filename, 'a') as f:
        # For each bounding box, label, and score
            for b, l, s in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            # Convert the bounding box, label, and score to string
                line = f"{IMG['file_name']}, {b}, {l}, {s}\n"
                # Write the line to the file
                f.write(line)
        source_img.save(IMG['file_name']+'aerialagain.png', "png")
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        print("Outputs",results)
        print("image done")
print("Processing done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Omni-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
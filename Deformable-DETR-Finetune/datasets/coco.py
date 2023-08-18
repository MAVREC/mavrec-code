# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, image_set, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file, image_set,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
     # visDrone
    print("rgs.dataset_file: ", args.dataset_file)
    if args.dataset_file == "visDrone":
        PATHS = {
            # MOT VAL/TEST SWAP
            "train": (root / "VisDrone2019-DET-train/images", root / 'annotations_visdrone_train'),
            "val": (root / "VisDrone2019-DET-val/images", root / 'annotations_visdrone_val.json'),
           # "test": (root / "scaled_label_test-dev", root / 'scaled_visdrone_mot_test-dev.json'),

            # # MOT  
            # "train": (root / "scaled_label_train", root / 'scaled_visdron_mot_train.json'),
            # "val": (root / "scaled_label_val", root / 'scaled_visdrone_mot_val.json'),
            # "test": (root / "scaled_label_test-dev", root / 'scaled_visdrone_mot_test-dev.json'),

            # DET
            # "train": (root / "scaled_label_train", root / "annotations" / 'scaled_annotations_visdrone_train.json'),
            # "val": (root / "scaled_label_val", root / "annotations" / 'scaled_annotations_visdrone_val.json'),
            # "test": (root / "VisDrone2019-DET-test-dev/images", root / "annotations" / 'annotations_visdrone_dev.json'),
          
        }   
    # GAAMOD
    elif args.dataset_file == "gamod":
        PATHS = {
            # "train": (root / "scaled_dataset"/ "all_ground_images", root / 'supervised_annotations' / 'ground' / 'aligned_ids' / 'ground_train_aligned_ids.json'),
            # "val": (root / "scaled_dataset" / "all_ground_images", root / 'supervised_annotations' / 'ground' / 'aligned_ids' / 'ground_valid_aligned_ids.json'),
            #"test": (root / "images", root / 'train'/'scaled_visdrone_mot_test-dev.json'),

            # ground
            # "train": (root / "scaled_dataset"/ "all_ground_images", root / 'supervised_annotations' / 'merged'  / 'aerial_ground_train_aligned_ids_w_indicator.json'),

            # "val": (root / "scaled_dataset" / "all_ground_images", root / 'supervised_annotations' / 'merged'  / 'aerial_ground_valid_aligned_ids_w_indicator.json'),

            # "test": (root / "scaled_dataset" / "all_ground_images", root / 'supervised_annotations' / 'merged'  / 'aerial_ground_valid_aligned_ids_w_indicator.json'),

            # aerial
            "train": (root / 'scaled_dataset' / 'train' / 'droneview', root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_train_aligned_ids_w_indicator.json'),

            "val": (root / 'scaled_dataset' / 'val' / 'droneview', root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_valid_aligned_ids_w_indicator.json'),

            "test": (root / 'scaled_dataset' / 'test' / 'droneview', root / 'supervised_annotations' / 'aerial' / 'aligned_ids' / 'aerial_test_aligned_ids.json'),

        }   
    else:
        PATHS = {
            "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        }

    print("PATHS: ", PATHS)
    img_folder, ann_file = PATHS[image_set]
    # print("IMAGE FOLDER: ", img_folder, "ANN_FILE: ", ann_file)

    dataset = CocoDetection(img_folder, ann_file, image_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset

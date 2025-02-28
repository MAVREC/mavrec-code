
import torch
import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'car','N/A', 'other','N/A', 'van','N/A', 'truck', 'N/A','tram', 'N/A','person', 'N/A','streetlight', 'N/A','trafficlight', 'N/A','bicycle', 'N/A','bus','N/A'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]




# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device = 'cuda:0')
    return b

def plot_results(pil_img, prob, boxes, name):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
    plt.axis('off')
    plt.savefig(name)
    plt.show()


def plot_sample_location_point(image_dir, sample_location_point, spatial_shape) :
  image = Image.open(image_dir)
  plt.imshow(image)
  h = image.shape[0]
  w = image.shape[1]
  h_= spatial_shape[0]
  w = spatial_shape[1]
  for i in range(300):
    x = sample_location_point[i,0]*h_*(h/h_)
    y = sample_location_point[i,1]*w_*(w/w_)
    plt.plot(sample_location_point[i,0], sample_location_point[i,1], 'o', color='black');
  plt.savefig('abcd.png')
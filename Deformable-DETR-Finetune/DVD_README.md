# DVD README
This readme explain how to setup the code-base used for trainining DVD benchmarks. 

## Installation 
Please refer to Deformable DETR's [github page](https://github.com/fundamentalvision/Deformable-DETR) for dependencies and other installations. We do not rely on anything additional.

## Dataset
We maintain distinct directories for the two views: Ground and Aerial samples. Our approach involves extracting samples from individual annotation files, which are then utilized to generate a combined annotation file encompassing both views. Instead of adding the directory path as a prefix to each image beforehand, we dynamically include it during training. We can do that because the image's file names describes the view. This requires a few steps seen below. Everything else is, as in the orignal code base. 


1. Generate a COCO annotation file containing both aerial and ground samples. File paths should only contain the base name. 
2. In following file: ```datasets/torchvision_datasets/coco.py``` change the ```__get_item__```  function's variables in the indentation below  ```if self.imageset == 'train'```. 

Thats it. Run everything else as normal. 
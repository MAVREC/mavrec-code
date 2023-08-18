# DVD README
This readme explain how to setup the code-base used for trainining DVD benchmarks. 

## Installation 
Please refer to YOLOv7's [github page](https://github.com/WongKinYiu/yolov7) for dependencies and other installations. We do not rely on anything additional onces.

## Dataset
We maintain distinct directories for the two views: Ground and Aerial samples. Our approach involves extracting samples from individual annotation files, which are then utilized to generate a combined annotation file encompassing both views. Instead of adding the directory path as a prefix to each image beforehand, we dynamically include it during training. We can do that because the image's file names describes the view. This requires a few steps seen below. Everything else is, as in the orignal code base. 


1. Generate a yolo annotations files containing both aerial and ground samples. File paths should only contain the base name. We just converted directly from a COCO file.
2. In following file: ```utils/touched_datasets.py``` change the ```label2img_paths()``` function's variables in the indentation below  ```if self.imageset == 'train'```. 

Thats it. Run everything else as normal. 



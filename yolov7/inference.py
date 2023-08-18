import torch
from torchvision import transforms
from PIL import Image
from models import Darknet
from utils.utils import non_max_suppression, rescale_boxes
from utils.datasets import letterbox

def load_model():
    # Load YOLOv7 model
    model = Darknet("cfg/yolov7.cfg")
    model.load_weights("weights/yolov7.weights")
    model.eval()
    return model

def preprocess_image(image):
    # Preprocess image
    img_size = 416
    img = letterbox(image, new_shape=img_size)[0]
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def postprocess_output(output, img_shape, conf_thres=0.5, nms_thres=0.4):
    # Apply post-processing to YOLOv7 output
    output = non_max_suppression(output, conf_thres, nms_thres)[0]
    if output is not None:
        output = rescale_boxes(output, img_shape, (416, 416))
    return output

def run_inference(image_path, output_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = preprocess_image(image)
    
    # Load YOLOv7 model
    model = load_model()
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Post-process output
    img_shape = image.size[::-1]
    output = postprocess_output(output, img_shape)
    
    # Generate MSCOCO formatted annotations file
    annotations = []
    if output is not None:
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in output:
            annotation = {
                "image_id": 1,  # Assuming only one image is processed
                "category_id": int(cls_pred),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf * cls_conf),
                'area': float(x2 - x1) * float(y2 - y1),
                "iscrowd": 0,
                "segmentation": []
            }
            annotations.append(annotation)
    
    # Save annotations to file
    with open(output_path, 'w') as f:
        for annotation in annotations:
            f.write(str(annotation) + '\n')
    
    print("Inference completed. Annotations saved to:", output_path)

# Run inference on an image and save the annotations
image_path = "path/to/your/image.jpg"
output_path = "path/to/your/output.txt"
run_inference(image_path, output_path)

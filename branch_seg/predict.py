import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image,ImageOps
from skimage.morphology import skeletonize

from blueberry_seg.runtime import init_runtime,project_path
from .evaluate import probability_map
from .model import load_model,resolve_device


def main():
    parser = argparse.ArgumentParser(description='Predict branch mask and 2D skeleton on a new RGB image')
    parser.add_argument('--weights',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',default='runs/branch/prediction')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threshold',type=float,default=0.5)
    args = parser.parse_args()
    if not 0<args.threshold<1:
        parser.error('threshold must be between zero and one')
    init_runtime(4)
    selected_device = resolve_device(args.device)
    model,checkpoint = load_model(project_path(args.weights),selected_device)
    image = Image.open(project_path(args.source)).convert('RGB')
    # Preserve aspect ratio exactly as in the prepared training set.
    fitted = ImageOps.contain(image,(640,640))
    offset = ((640-fitted.width)//2,(640-fitted.height)//2)
    canvas = Image.new('RGB',(640,640),(114,114,114))
    canvas.paste(fitted,offset)
    probability = probability_map(model,canvas,checkpoint['config']['imgsz'],selected_device)
    x,y = offset
    probability = probability[y:y+fitted.height,x:x+fitted.width]
    probability = cv2.resize(probability,image.size,interpolation=cv2.INTER_LINEAR)
    prediction = probability>=args.threshold
    skeleton = skeletonize(prediction)
    directory = project_path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8)*255).save(directory/'branch_mask.png')
    Image.fromarray(skeleton.astype(np.uint8)*255).save(directory/'branch_skeleton.png')
    np.savez_compressed(directory/'branch_probability.npz',probability=probability)
    pixels = np.array(image)
    pixels[prediction] = (pixels[prediction]*0.45+np.array([255,75,20])*0.55).astype(np.uint8)
    Image.fromarray(pixels).save(directory/'branch_overlay.jpg',quality=95)
    record = {'source':args.source,'weights':args.weights,'threshold':args.threshold,
              'image_size':image.size,'coordinate_system':'original RGB pixels',
              'prediction_pixels':int(prediction.sum()),'skeleton_pixels':int(skeleton.sum()),
              'fruit_branch_association_verified':False,'three_dimensional_geometry':False,
              'note':'2D semantic prediction and skeleton only; not a robot grasp command.'}
    (directory/'prediction.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

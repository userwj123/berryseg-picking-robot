import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

from blueberry_seg.runtime import init_runtime, project_path
from .data import normalized_tensor,validate_manifest
from .metrics import confusion,scores
from .model import load_model,resolve_device


@torch.inference_mode()
def probability_map(model,image,imgsz,selected_device):
    inputs = normalized_tensor(image.resize((imgsz,imgsz),Image.Resampling.BILINEAR))[None].to(selected_device)
    probability = model(inputs)['out'].softmax(1)[0,1].cpu().numpy()
    return cv2.resize(probability,image.size,interpolation=cv2.INTER_LINEAR)


def panel(image,label,prediction,path,title):
    panels = []
    pixels = np.array(image)
    for foreground,caption in ((label==1,'AI-assisted annotation'),(prediction,'Prediction (entire image)'),(skeletonize(prediction),'Prediction skeleton')):
        overlay = pixels.copy()
        overlay[label==255] = (overlay[label==255]*0.65).astype(np.uint8)
        overlay[foreground] = (overlay[foreground]*0.3+np.array([255,75,20])*0.7).astype(np.uint8)
        picture = Image.fromarray(overlay)
        picture.thumbnail((400,400))
        canvas = Image.new('RGB',(400,430),'white')
        canvas.paste(picture,((400-picture.width)//2,30))
        ImageDraw.Draw(canvas).text((8,8),caption,fill='black')
        panels.append(canvas)
    result = Image.new('RGB',(1200,460),'white')
    ImageDraw.Draw(result).text((8,5),title,fill='black')
    for index,picture in enumerate(panels):
        result.paste(picture,(index*400,28))
    result.save(path,quality=95)


def evaluate(weights,data,output,selected_device='cpu'):
    directory,output = Path(data),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records,counts = validate_manifest(directory,allow_assisted=True)
    model,checkpoint = load_model(weights,selected_device)
    imgsz = checkpoint['config']['imgsz']
    # Threshold is selected only on val; test is evaluated once with that threshold.
    thresholds = [0.25,0.35,0.5,0.65,0.75]
    maps = {}
    for record in records:
        if record['split']=='val':
            with Image.open(directory/record['prepared_image']) as im:
                maps[record['id']] = probability_map(model,im.convert('RGB'),imgsz,selected_device)
    search = []
    for threshold in thresholds:
        total = np.zeros(4,dtype=np.int64)
        for record in records:
            if record['split']=='val':
                target = np.asarray(Image.open(directory/record['mask']))
                total += confusion(maps[record['id']]>=threshold,target)
        search.append({'threshold':threshold,**scores(total)})
    best = max(search,key=lambda r:r['dice'])
    threshold = best['threshold']
    result = {'architecture':checkpoint['architecture'],'best_epoch':checkpoint['epoch'],
              'imgsz':imgsz,'threshold':threshold,'threshold_selection':'val Dice only',
              'validation_threshold_search':search,'sets':{},
              'label_provenance':'AI-assisted local branch masks; no expert human review',
              'evaluation_scope':'Only labels 0/1 are scored; 255 and uncertain boundaries are ignored. Not full-image accuracy, fruit-branch association, 3D accuracy or gripping success.'}
    for split in ('val','test'):
        base = output/split
        base.mkdir(exist_ok=True)
        rows,total = [],np.zeros(4,dtype=np.int64)
        for record in records:
            if record['split']!=split:
                continue
            image = Image.open(directory/record['prepared_image']).convert('RGB')
            label = np.asarray(Image.open(directory/record['mask']))
            probability = maps.get(record['id'])
            if probability is None:
                probability = probability_map(model,image,imgsz,selected_device)
            prediction = probability>=threshold
            row_counts = confusion(prediction,label)
            total += row_counts
            row = {'id':record['id'],**scores(row_counts),
                   'ignore_fraction':float(np.mean(label==255)),
                   'prediction_pixels_in_ignored_region':int(np.sum(prediction&(label==255))),
                   'predicted_components':int(cv2.connectedComponents(prediction.astype(np.uint8))[0]-1)}
            rows.append(row)
            Image.fromarray(prediction.astype(np.uint8)*255).save(base/f'{record["id"]}_prediction.png')
            Image.fromarray(skeletonize(prediction).astype(np.uint8)*255).save(base/f'{record["id"]}_skeleton.png')
            np.savez_compressed(base/f'{record["id"]}_probability.npz',probability=probability)
            panel(image,label,prediction,base/f'{record["id"]}_comparison.jpg',
                  f'{record["id"]} | {split} | labeled-pixel IoU {row["iou"]:.3f} Dice {row["dice"]:.3f} | threshold {threshold}')
        result['sets'][split] = {'images':counts[split],'micro':scores(total),
                                'macro_iou':float(np.mean([r['iou'] for r in rows])),
                                'macro_dice':float(np.mean([r['dice'] for r in rows])),
                                'per_image':rows}
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',required=True)
    parser.add_argument('--data',default='data/processed/branch_pilot_v1')
    parser.add_argument('--output',default='reports/branch_pilot/evaluation')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    init_runtime(4)
    result = evaluate(project_path(args.weights),project_path(args.data),project_path(args.output),resolve_device(args.device))
    print(json.dumps({split:row['micro'] for split,row in result['sets'].items()},indent=2))


if __name__=='__main__':
    main()

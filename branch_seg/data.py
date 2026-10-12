import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485,0.456,0.406])[:,None,None]
STD = torch.tensor([0.229,0.224,0.225])[:,None,None]


def normalized_tensor(image):
    pixels = np.array(image,dtype=np.float32)/255
    return (torch.from_numpy(pixels).permute(2,0,1)-MEAN)/STD


def validate_manifest(directory, allow_assisted=False):
    directory = Path(directory)
    records = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if not records or {r['split'] for r in records} != {'train','val','test'}:
        raise ValueError('Dataset needs train/val/test records')
    groups, pixels, counts = {}, {}, {}
    for record in records:
        if not record.get('human_reviewed') and not allow_assisted:
            raise ValueError('AI-assisted labels require explicit allow_assisted_labels=true')
        if not record.get('ai_visual_reviewed'):
            raise ValueError(f'Visual review missing: {record["id"]}')
        split = record['split']
        for mapping,key in ((groups,record['group']),(pixels,record['prepared_sha256'])):
            if key in mapping and mapping[key] != split:
                raise ValueError(f'Cross-split duplicate/source group: {key}')
            mapping[key] = split
        with Image.open(directory/record['prepared_image']) as image, Image.open(directory/record['mask']) as mask:
            if image.size != mask.size:
                raise ValueError('Image/mask size mismatch')
            if hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()!=record['prepared_sha256']:
                raise ValueError('Image changed after source/duplicate audit')
            array = np.asarray(mask)
        if not set(np.unique(array)).issubset({0,1,255}):
            raise ValueError('Mask must encode background=0, branch=1, ignore=255')
        if (array==1).sum()<30 or (array==0).sum()<100:
            raise ValueError('Mask lacks positive/background supervision')
        if hashlib.sha256(array.tobytes()).hexdigest() != record['mask_sha256']:
            raise ValueError('Mask changed after review; update the review manifest')
        counts[split] = counts.get(split,0)+1
    return records, counts


class BranchDataset(Dataset):
    def __init__(self,directory,records,split,imgsz=512,augment=False):
        self.directory = Path(directory)
        self.records = [r for r in records if r['split']==split]
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        with Image.open(self.directory/record['prepared_image']) as im:
            image = im.convert('RGB').resize((self.imgsz,self.imgsz),Image.Resampling.BILINEAR)
        with Image.open(self.directory/record['mask']) as im:
            mask = im.convert('L').resize((self.imgsz,self.imgsz),Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<0.5:
                image,mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random()<0.25:
                image,mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM),mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85,1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.9,1.1))
        return normalized_tensor(image),torch.from_numpy(np.array(mask,dtype=np.int64)),record['id']

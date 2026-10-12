"""Import public COCO / YOLO instance polygons without semantic-mask recovery."""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .prepare import preview
from .runtime import ROOT

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def polygon_xy(values, width=1, height=1):
    xy = np.asarray(values, dtype=float)
    if xy.size < 6 or xy.size % 2:
        raise ValueError('Polygon needs at least 3 XY pairs')
    xy = xy.reshape(-1, 2) / [width, height]
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite polygon')
    # Permit only tiny numeric rounding at the border, not silently wrong geometry.
    if xy.min() < -1e-6 or xy.max() > 1+1e-6:
        raise ValueError('Polygon outside image')
    xy = np.clip(xy, 0, 1)
    area = abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))-np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))/2
    if len(np.unique(xy, axis=0)) < 3 or area <= 1e-12:
        raise ValueError('Degenerate polygon')
    return xy.tolist()


def read_yolo(path, names):
    instances = []
    for index, row in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
        if not row.strip():
            continue
        parts = row.split()
        category = int(parts[0])
        if category not in names:
            raise ValueError(f'Unknown category {category}')
        instances.append({'id': index, 'category': names[category], 'polygon': polygon_xy(parts[1:])})
    return instances


def source_group(stem):
    # Roboflow hashes and augmented tiles are variants of the same source photo.
    base = stem.split('.rf.')[0]
    base = re.sub(r'_(jpg|jpeg|png)$', '', base, flags=re.I)
    base = re.sub(r'_\d+_\d+$', '', base)
    base = re.sub(r'_(jpg|jpeg|png)$', '', base, flags=re.I)
    # Hold out all camera views of the same explicitly named plant/photo family.
    base = re.sub(r'^(left|right|top)_', '', base, flags=re.I)
    return 'bsail:'+base.casefold()


def coco_records(raw):
    roots = list(raw.rglob('instances_train2017.json'))
    if len(roots) != 1:
        raise ValueError('Expected one COCO annotation root')
    base = roots[0].parent.parent
    records, quarantined = [], []
    for split, suffix in [('train', 'train2017'), ('val', 'val2017'), ('unlabeled', 'test2017')]:
        annotation_path = base/'annotations'/f'instances_{suffix}.json'
        document = json.loads(annotation_path.read_text(encoding='utf-8'))
        categories = {c['id']: c['name'] for c in document['categories']}
        by_image = defaultdict(list)
        for annotation in document['annotations']:
            by_image[annotation['image_id']].append(annotation)
        image_ids = {im['id'] for im in document['images']}
        if set(by_image)-image_ids:
            raise ValueError('Annotation refers to an unknown image')
        annotation_ids = [a['id'] for a in document['annotations']]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError('Duplicate COCO annotation IDs')
        for image in document['images']:
            filename = Path(image['file_name']).name
            matches = [p for p in (base/suffix/filename, base/image['file_name']) if p.is_file()]
            if not matches:
                raise FileNotFoundError(f'COCO image missing: {filename}')
            source = matches[0]
            record = {'dataset': 'ni2020', 'source_image': str(source.relative_to(ROOT)),
                      'source_annotation': str(annotation_path.relative_to(ROOT)),
                      'original_split': split, 'split': split, 'group': f'ni2020:{filename}',
                      'source_image_id': image['id'], 'label_provenance': 'published_coco_instance_polygons'}
            try:
                with Image.open(source) as im:
                    if im.size != (image['width'], image['height']):
                        raise ValueError('COCO dimensions differ from image')
                instances = []
                for ann in by_image[image['id']]:
                    if ann.get('iscrowd', 0):
                        raise ValueError('Crowd annotation needs an ignore-region policy')
                    segmentation = ann['segmentation']
                    if not isinstance(segmentation, list) or len(segmentation) != 1:
                        raise ValueError('Multipart/RLE instance cannot be represented losslessly as one YOLO polygon')
                    instances.append({'id': ann['id'], 'category': categories[ann['category_id']],
                                      'polygon': polygon_xy(segmentation[0], image['width'], image['height'])})
                record['instances'] = instances
                records.append(record)
            except (ValueError, OSError, KeyError) as exc:
                quarantined.append({**record, 'reason': str(exc)})
    return records, quarantined


def bsail_records(raw):
    roots = [p.parent for p in raw.rglob('data.yaml') if (p.parent/'train/images').is_dir()]
    if len(roots) != 1:
        raise ValueError(f'Expected one BSAIL training export, found {len(roots)}')
    base = roots[0]
    document = yaml.safe_load((base/'data.yaml').read_text(encoding='utf-8'))
    raw_names = document['names']
    names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else {int(k): v for k,v in raw_names.items()}
    partitions = [(base/'train', 'train'), (base/'valid', 'val')]
    manual_tests = [p for p in raw.rglob('test') if p.is_dir() and (p/'images').is_dir() and (p/'labels').is_dir()]
    if len(manual_tests) > 1:
        raise ValueError('Ambiguous manual test export')
    partitions += [(p, 'test') for p in manual_tests]
    records, quarantined = [], []
    for directory, split in partitions:
        partition_names = names
        if split == 'test':
            test_yaml = directory.parent/'data.yaml'
            if not test_yaml.exists():
                raise ValueError('Manual test category mapping is missing')
            test_names = yaml.safe_load(test_yaml.read_text(encoding='utf-8'))['names']
            partition_names = dict(enumerate(test_names)) if isinstance(test_names, list) else {int(k):v for k,v in test_names.items()}
        images = sorted(p for p in (directory/'images').iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        image_stems = {p.stem for p in images}
        if len(images) != len(image_stems):
            raise ValueError('Duplicate image stems')
        orphan_labels = {p.stem for p in (directory/'labels').glob('*.txt')} - image_stems
        if orphan_labels:
            raise ValueError(f'Orphan labels: {len(orphan_labels)}')
        for source in images:
            label = directory/'labels'/f'{source.stem}.txt'
            record = {'dataset': 'bsail_sam', 'source_image': str(source.relative_to(ROOT)),
                      'source_annotation': str(label.relative_to(ROOT)), 'split': split, 'original_split': split,
                      'group': source_group(source.stem),
                      'label_provenance': 'published_manual_test_polygons' if split == 'test' else 'published_sam_pseudo_instances'}
            try:
                if not label.exists():
                    raise ValueError('Missing label is not assumed to mean background')
                record['instances'] = read_yolo(label, partition_names)
                records.append(record)
            except (ValueError, OSError) as exc:
                quarantined.append({**record, 'reason': str(exc)})
    return records, quarantined


def audit_records(records, quarantined):
    """Keep held-out split priority; quarantine duplicates and source variants."""
    priority = {'test': 0, 'val': 1, 'train': 2, 'unlabeled': 3}
    verified = []
    for index, record in enumerate(records):
        try:
            source = ROOT/record['source_image']
            with Image.open(source) as im:
                rgb = im.convert('RGB')
                record['source_size'] = list(rgb.size)
                digest = hashlib.sha256()
                digest.update(str(rgb.size).encode())
                digest.update(rgb.tobytes())
                record['pixel_sha256'] = digest.hexdigest()
            record['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            verified.append(record)
            if (index+1) % 200 == 0:
                print(f'Image audit: {index+1}/{len(records)}', flush=True)
        except (OSError, ValueError) as exc:
            quarantined.append({**record, 'reason': f'Image decode: {exc}'})
    group_splits = defaultdict(set)
    for record in verified:
        group_splits[record['group']].add(record['split'])
    cross_groups = {g: sorted(s) for g,s in group_splits.items() if len(s)>1}
    retained, seen = [], {}
    for record in sorted(verified, key=lambda r: (priority[r['split']], r['source_image'])):
        best = min(group_splits[record['group']], key=lambda s: priority[s])
        reason = None
        if record['split'] != best:
            reason = f'Source photo variants also occur in held-out {best} split'
        elif record['pixel_sha256'] in seen:
            reason = f'Exact decoded-image duplicate of {seen[record["pixel_sha256"]]}'
        if reason:
            quarantined.append({**record, 'reason': reason})
        else:
            seen[record['pixel_sha256']] = record['source_image']
            retained.append(record)
    return retained, cross_groups


def category_names(records):
    return sorted({i['category'] for r in records for i in r['instances']})


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def export_dataset(records, output, class_mode, max_side=1280, cached=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    categories = category_names(records) if class_mode == 'maturity' else ['blueberry']
    mapping = {name: index for index, name in enumerate(categories)}
    manifests = []
    cache = cached if cached is not None else {}
    review_count = Counter()
    for index, record in enumerate(records):
        token = record['dataset']+'_'+hashlib.sha256(record['source_image'].encode()).hexdigest()[:20]
        image_rel = f'images/{record["split"]}/{token}.jpg'
        target = output/image_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT/record['source_image']
        if record['source_image'] in cache:
            link_or_copy(cache[record['source_image']], target)
        else:
            with Image.open(source) as im:
                rgb = im.convert('RGB')
                if max_side:
                    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                rgb.save(target, quality=95)
            cache[record['source_image']] = target
        result = {k: v for k,v in record.items() if k != 'instances'}
        result['image'] = image_rel
        result['instance_count'] = len(record['instances'])
        result['annotation_ids'] = [i['id'] for i in record['instances']]
        result['category_counts'] = dict(Counter(i['category'] for i in record['instances']))
        if record['split'] != 'unlabeled':
            label_rel = f'labels/{record["split"]}/{token}.txt'
            label_path = output/label_rel
            label_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for instance in record['instances']:
                category = mapping[instance['category']] if class_mode == 'maturity' else 0
                coords = np.asarray(instance['polygon']).reshape(-1)
                rows.append(str(category)+' '+' '.join(f'{v:.9f}' for v in coords))
            label_path.write_text('\n'.join(rows)+('\n' if rows else ''), encoding='utf-8')
            result['label'] = label_rel
            if review_count[record['split']] < 3:
                review = output/'review'
                review.mkdir(exist_ok=True)
                with Image.open(target) as image:
                    polys = [np.asarray(i['polygon'])*image.size for i in record['instances']]
                    preview(image, polys, review/f'{token}.jpg', f'{record["dataset"]} {record["split"]} | {len(polys)} instances')
                review_count[record['split']] += 1
        manifests.append(result)
        if (index+1) % 200 == 0:
            print(f'{output.name}: exported {index+1}/{len(records)}', flush=True)
    data = {'train': 'images/train', 'val': 'images/val', 'names': dict(enumerate(categories))}
    if any(r['split'] == 'test' for r in records):
        data['test'] = 'images/test'
    if not all(any(r['split'] == s for r in records) for s in ('train', 'val')):
        raise ValueError('Need nonempty training and validation splits')
    (output/'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    rng = random.Random(42)
    for split, count in [('train', 8), ('val', 4)]:
        subset = [r for r in manifests if r['split'] == split and r['instance_count']]
        rng.shuffle(subset)
        (output/f'{split}_smoke.txt').write_text(''.join('./'+r['image']+'\n' for r in subset[:count]), encoding='utf-8')
    smoke = {**data, 'train': 'train_smoke.txt', 'val': 'val_smoke.txt'}
    (output/'dataset_smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False), encoding='utf-8')
    write_json(output/'manifest.json', manifests)
    summary = {'format': 'yolo_instance_polygons', 'class_mode': class_mode, 'names': categories,
               'image_counts': dict(Counter(r['split'] for r in manifests)),
               'instance_counts': {s: sum(r['instance_count'] for r in manifests if r['split']==s) for s in ('train','val','test')},
               'provenance_counts': dict(Counter(r['label_provenance'] for r in manifests)),
               'max_side': max_side, 'quarantine_report': '../../../reports/dataset_download/import_audit.json',
               'note': 'Published instance polygons; SAM-generated train/val labels remain pseudo-labels. Unlabeled images excluded from YAML.'}
    write_json(output/'summary.json', summary)
    print(output.name, json.dumps(summary, ensure_ascii=False), flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed')
    parser.add_argument('--max-side', type=int, default=1280)
    args = parser.parse_args()
    if args.max_side < 0:
        parser.error('max-side must be nonnegative')
    quarantined = []
    print('Reading COCO instances', flush=True)
    ni, rejected = coco_records(args.raw/'ni2020/extracted')
    quarantined += rejected
    print('Reading BSAIL instance polygons', flush=True)
    bsail, rejected = bsail_records(args.raw/'bsail_sam/extracted')
    quarantined += rejected
    schema_valid_counts = dict(Counter((r['dataset']+':'+r['split']) for r in ni+bsail))
    raw_counts = dict(Counter((r['dataset']+':'+r['split']) for r in ni+bsail+quarantined))
    print('Auditing source images and duplicates', flush=True)
    records, cross_groups = audit_records(ni+bsail, quarantined)
    cache = {}
    for dataset in ('ni2020', 'bsail_sam'):
        subset = [r for r in records if r['dataset'] == dataset]
        cache = export_dataset(subset, args.output/dataset, 'single', args.max_side, cache)
        cache = export_dataset(subset, args.output/f'{dataset}_maturity', 'maturity', args.max_side, cache)
    export_dataset([r for r in records if r['split'] != 'unlabeled'], args.output/'public_combined', 'single', args.max_side, cache)
    audit = {'original_record_counts': raw_counts, 'schema_valid_record_counts': schema_valid_counts,
             'retained_counts': dict(Counter(r['dataset']+':'+r['split'] for r in records)),
             'quarantined_images': len(quarantined), 'quarantine_reasons': dict(Counter(r['reason'] for r in quarantined)),
             'cross_split_source_groups': cross_groups, 'quarantine': quarantined,
             'policy': 'Keep held-out priority test > val > train; never fill missing labels as background. Filename and decoded-pixel checks do not prove video/plant independence.'}
    report = ROOT/'reports/dataset_download/import_audit.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, audit)
    print('Import complete; audit:', report, flush=True)

if __name__ == '__main__':
    main()

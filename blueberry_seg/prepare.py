import argparse
import hashlib
import io
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .labels import binary_mask, instance_polygons, yolo_rows

SOURCE_PATTERN = re.compile(r'Camera\s*(\d+)\s*Video\s*\((\d+)\)_(\d+)', re.I)


def read_image(path):
    # Short-lived reads also handle transient removable-disk I/O without ignoring bad files.
    for attempt in range(3):
        try:
            with Image.open(io.BytesIO(path.read_bytes())) as image:
                return image.copy()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.25*(attempt+1))


def source_group(stem):
    match = SOURCE_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f'Cannot derive camera/video group from filename: {stem}')
    camera, video, _ = match.groups()
    return f'camera_{camera}_video_{video}'


def split_groups(stems, seed=42, val_ratio=0.15, test_ratio=0.15):
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio+test_ratio >= 1:
        raise ValueError('val/test ratios must be positive and sum to less than 1')
    groups_by_camera = defaultdict(set)
    for stem in stems:
        match = SOURCE_PATTERN.fullmatch(stem)
        if not match:
            raise ValueError(f'Unrecognized source name: {stem}')
        groups_by_camera[match.group(1)].add(source_group(stem))
    rng = random.Random(seed)
    assignments = {}
    for camera in sorted(groups_by_camera):
        groups = sorted(groups_by_camera[camera])
        if len(groups) < 3:
            raise ValueError(f'Camera {camera} needs at least 3 videos for disjoint splits')
        rng.shuffle(groups)
        n_val = max(1, round(len(groups)*val_ratio))
        n_test = max(1, round(len(groups)*test_ratio))
        if n_val+n_test >= len(groups):
            raise ValueError('Not enough video groups for requested ratios')
        for i, group in enumerate(groups):
            assignments[group] = 'val' if i < n_val else 'test' if i < n_val+n_test else 'train'
    return {stem: assignments[source_group(stem)] for stem in stems}


def preview(image, polygons, path, caption):
    image = image.copy().convert('RGB')
    scale = min(1, 800/max(image.size))
    image = image.resize((round(image.width*scale), round(image.height*scale)))
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    rng = random.Random(42)
    for i, polygon in enumerate(polygons):
        points = [tuple(p) for p in polygon*scale]
        color = tuple(rng.randrange(50, 245) for _ in range(3))
        draw.polygon(points, fill=(*color, 75), outline=(*color, 255), width=2)
        draw.text(points[0], str(i+1), fill=(255, 255, 255, 255))
    image = Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')
    canvas = Image.new('RGB', (image.width, image.height+30), 'white')
    canvas.paste(image, (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), caption, fill='black')
    canvas.save(path, quality=92)


def main():
    parser = argparse.ArgumentParser(description='Convert DB-1 binary masks to auditable YOLO pseudo-instances')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/processed/db1'))
    parser.add_argument('--method', choices=['components', 'watershed'], default='components')
    parser.add_argument('--min-area', type=int, default=30, help='Source-resolution pixel area')
    parser.add_argument('--max-side', type=int, default=1280, help='Export size; 0 keeps original dimensions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--smoke-train', type=int, default=16)
    parser.add_argument('--smoke-val', type=int, default=4)
    args = parser.parse_args()
    if args.min_area < 1 or args.max_side < 0 or min(args.smoke_train, args.smoke_val) < 1:
        parser.error('Invalid area, size or smoke subset count')
    if not args.source.is_dir():
        parser.error(f'Source not found: {args.source}')
    # Never silently overwrite a converted dataset with changed split/geometry settings.
    if args.output.exists() and any(p.is_file() for p in args.output.rglob('*')):
        parser.error('Output is not empty; choose a new output folder to preserve provenance')
    images = {p.stem: p for p in (args.source/'Images').iterdir() if p.suffix.lower() in ('.bmp', '.jpg', '.jpeg', '.png')}
    masks = {p.stem: p for p in (args.source/'Masks').glob('*.png')}
    if set(images) != set(masks):
        raise ValueError(f'Unpaired files: images-only={len(set(images)-set(masks))}, masks-only={len(set(masks)-set(images))}')
    stems = sorted(images)
    split = split_groups(stems, args.seed, args.val_ratio, args.test_ratio)
    for name in ('train', 'val', 'test'):
        (args.output/'images'/name).mkdir(parents=True, exist_ok=True)
        (args.output/'labels'/name).mkdir(parents=True, exist_ok=True)
    (args.output/'review').mkdir(exist_ok=True)
    records = []
    totals = Counter()
    preview_indices = set(np.linspace(0, len(stems)-1, min(12, len(stems)), dtype=int).tolist())
    for index, stem in enumerate(stems):
        try:
            image = read_image(images[stem]).convert('RGB')
        except OSError as exc:
            raise OSError(f'Cannot read source image: {images[stem]}') from exc
        mask = binary_mask(np.asarray(read_image(masks[stem])))
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f'Image/mask dimensions differ: {stem}')
        polygons, info = instance_polygons(mask, args.method, args.min_area)
        rows = yolo_rows(polygons, image.width, image.height)
        # ASCII file names avoid loader issues on Windows and preserve exact source in manifest.
        name = SOURCE_PATTERN.fullmatch(stem)
        export_name = f'c{name[1]}_v{name[2]}_f{name[3]}'
        image_path = args.output/'images'/split[stem]/f'{export_name}.jpg'
        if args.max_side and max(image.size) > args.max_side:
            image.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        image.save(image_path, quality=95)
        label_path = args.output/'labels'/split[stem]/f'{export_name}.txt'
        label_path.write_text('\n'.join(rows)+ ('\n' if rows else ''), encoding='utf-8')
        record = {'source_image': str(images[stem]), 'source_mask': str(masks[stem]),
                  'source_mask_sha256': hashlib.sha256(masks[stem].read_bytes()).hexdigest(),
                  'group': source_group(stem), 'split': split[stem],
                  'image': image_path.relative_to(args.output).as_posix(),
                  'label': label_path.relative_to(args.output).as_posix(), **info}
        records.append(record)
        totals.update(info)
        if index in preview_indices:
            original_size = (mask.shape[1], mask.shape[0])
            scaled_polygons = [p*np.array(image.size)/np.array(original_size) for p in polygons]
            preview(image, scaled_polygons, args.output/'review'/f'{export_name}.jpg',
                    f'{export_name} | {args.method} | {len(polygons)} derived labels')
        if (index+1) % 100 == 0:
            print(f'Converted {index+1}/{len(stems)}', flush=True)
    # No absolute root in these YAMLs: train.py binds their root to the YAML directory at runtime.
    data = {'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': {0: 'blueberry'}}
    (args.output/'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    rng = random.Random(args.seed)
    smoke_counts = {}
    for name, limit in [('train', args.smoke_train), ('val', args.smoke_val)]:
        subset = [r for r in records if r['split'] == name and r['derived_instances'] > 0]
        rng.shuffle(subset)
        subset = subset[:limit]
        if not subset:
            raise ValueError(f'Smoke {name} split has no positive images')
        # Relative list entries resolve against the list file, including after GPU migration.
        (args.output/f'{name}_smoke.txt').write_text(''.join('./'+r['image']+'\n' for r in subset), encoding='utf-8')
        smoke_counts[name] = len(subset)
    smoke_data = {**data, 'train': 'train_smoke.txt', 'val': 'val_smoke.txt'}
    (args.output/'dataset_smoke.yaml').write_text(yaml.safe_dump(smoke_data, sort_keys=False), encoding='utf-8')
    summary = {'source': str(args.source.resolve()), 'label_provenance': 'derived_from_binary_foreground_not_true_instance_ids',
               'method': args.method, 'min_area_source_pixels': args.min_area, 'max_side': args.max_side,
               'seed': args.seed, 'images': len(records), 'image_counts': dict(Counter(r['split'] for r in records)),
               'video_counts': {s: len({r['group'] for r in records if r['split'] == s}) for s in ('train','val','test')},
               'smoke_counts': smoke_counts, 'totals': dict(totals),
               'warning': 'Merged touching fruit and fragments need manual instance review; no ripeness labels.'}
    (args.output/'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()


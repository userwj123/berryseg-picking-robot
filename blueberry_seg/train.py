import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import yaml

from .runtime import ROOT, bind_dataset, device, init_runtime, project_path


def main():
    parser = argparse.ArgumentParser(description='Train blueberry instance segmentation on CPU or CUDA')
    parser.add_argument('--config', default='configs/cpu_smoke.yaml')
    parser.add_argument('--data', help='Override dataset YAML')
    parser.add_argument('--device', help='cpu, auto, 0, cuda:0, or 0,1')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imgsz', type=int)
    parser.add_argument('--model', help='Pretrained/custom .pt weights or explicit --scratch architecture')
    parser.add_argument('--name')
    parser.add_argument('--resume', help='Resume an interrupted last.pt training state')
    parser.add_argument('--scratch', action='store_true', help='Explicitly allow training from a model YAML')
    args = parser.parse_args()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8'))
    threads = int(cfg.pop('threads', 4))
    if threads < 1:
        parser.error('threads must be positive')
    init_runtime(threads)
    for key in ('data', 'device', 'epochs', 'batch', 'imgsz', 'model', 'name'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg['device'] = device(cfg.get('device', 'cpu'))
    cfg['data'] = str(bind_dataset(cfg['data']))
    model_path = project_path(args.resume or cfg.pop('model'))
    if not model_path.is_file():
        if args.scratch and model_path.suffix == '.yaml':
            model_path = Path(args.model or 'yolo11n-seg.yaml')
        else:
            raise FileNotFoundError(f'Model missing: {model_path}. Run scripts/download_weights.py first.')
    if model_path.suffix == '.yaml' and not args.scratch:
        parser.error('Model YAML needs --scratch; random initialization is never a silent fallback')
    cfg['project'] = str(project_path(cfg.get('project', 'runs/segment')))
    cfg['task'] = 'segment'
    cfg['max_det'] = cfg.get('max_det', 500)
    from ultralytics import YOLO
    model = YOLO(str(model_path), task='segment')
    if model.task != 'segment':
        raise ValueError('Checkpoint is not an instance segmentation model')
    if args.resume:
        # Ultralytics resumes checkpoint hyperparameters; only supported overrides are passed.
        result = model.train(resume=True, data=cfg['data'], device=cfg['device'], workers=cfg.get('workers', 0))
    else:
        result = model.train(**cfg)
    run = Path(model.trainer.save_dir)
    packages = {name: importlib.metadata.version(name) for name in ('ultralytics', 'torch', 'torchvision', 'numpy', 'opencv-python', 'scikit-image')}
    metadata = {'python': platform.python_version(), 'packages': packages, 'device': cfg['device'],
                'config': cfg, 'initial_model': str(model_path), 'cwd': os.getcwd(),
                'metrics': result.results_dict if result is not None else {},
                'note': 'Consult the dataset manifest for published, manual or pseudo-label provenance; training validation is not a deployment test.'}
    (run/'run_summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f'Training complete. Weights and metrics: {run}', flush=True)


if __name__ == '__main__':
    main()


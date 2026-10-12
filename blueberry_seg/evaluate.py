import argparse
import json

from .runtime import bind_dataset, device, init_runtime, project_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate held-out instance labels; see dataset manifest for provenance')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='data/processed/db1/dataset.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--name', default='evaluation')
    args = parser.parse_args()
    init_runtime()
    from ultralytics import YOLO
    weights = project_path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)
    metrics = YOLO(str(weights)).val(data=str(bind_dataset(args.data)), split=args.split,
                                   device=device(args.device), imgsz=args.imgsz, batch=args.batch,
                                   workers=0, max_det=500, project=str(project_path('runs/evaluate')), name=args.name)
    (metrics.save_dir/'evaluation.json').write_text(json.dumps(metrics.results_dict, indent=2), encoding='utf-8')
    print(metrics.results_dict)


if __name__ == '__main__':
    main()


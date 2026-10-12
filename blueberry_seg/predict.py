import argparse
import json
from pathlib import Path

from .runtime import device, init_runtime, project_path


def main():
    parser = argparse.ArgumentParser(description='Predict per-berry instance masks; coordinates refer to original image')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True, help='Image, directory, video, or camera index')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--max-det', type=int, default=500)
    parser.add_argument('--output', default='runs/predict')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    init_runtime(args.threads)
    from ultralytics import YOLO
    weights = project_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output = project_path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Prediction output is not empty; choose a new folder')
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights), task='segment')
    source = int(args.source) if args.source.isdigit() else str(project_path(args.source))
    stream = model.predict(source=source, device=device(args.device), imgsz=args.imgsz, conf=args.conf,
                           max_det=args.max_det, retina_masks=True, stream=True, verbose=False)
    count = 0
    with (output/'instances.jsonl').open('w', encoding='utf-8') as handle:
        for frame_index, result in enumerate(stream):
            instances = []
            if result.masks is not None:
                for i, polygon in enumerate(result.masks.xy):
                    instances.append({'class': int(result.boxes.cls[i]), 'name': result.names[int(result.boxes.cls[i])],
                                      'confidence': float(result.boxes.conf[i]),
                                      'bbox_xyxy': result.boxes.xyxy[i].cpu().tolist(),
                                      'polygon_xy': polygon.tolist()})
            record = {'source': str(result.path), 'frame_index': frame_index,
                      'image_height': result.orig_shape[0], 'image_width': result.orig_shape[1],
                      'coordinate_frame': 'original_rgb_pixels', 'instances': instances}
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            result.save(filename=str(output/f'{frame_index:06d}_{Path(result.path).stem}.jpg'))
            count += 1
    print(f'Predicted {count} frames. Results: {output}')


if __name__ == '__main__':
    main()


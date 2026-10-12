"""Offline D435 replay: extraction, segmentation, depth geometry, and dry-run pose.

Outputs are candidate measurements and a labelled fictional robot transform.
This module neither imports the robot SDK nor connects to hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .rgbd import Recording, estimate_surface, project_depth, suction_rotation, transform_point
from .runtime import device, init_runtime, project_path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def imwrite(path, image):
    # Works with Windows Chinese paths.
    okay, encoded = cv2.imencode(path.suffix, image)
    if not okay:
        raise RuntimeError(f'Image encoding failed: {path}')
    encoded.tofile(str(path))


def contact_sheet(images, path, columns=4):
    tiles = []
    for label, bgr in images:
        tile = cv2.resize(bgr, (320, 240))
        cv2.rectangle(tile, (0, 0), (320, 25), (0, 0, 0), -1)
        cv2.putText(tile, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    imwrite(path, sheet)


def simulate_robot(target):
    """An explicit arithmetic example. All robot installation data is fictional."""
    from scipy.spatial.transform import Rotation
    T_B_F = np.eye(4)
    T_B_F[:3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
    T_B_F[:3, 3] = [-.20, -.10, .40]
    T_F_C = np.eye(4)
    T_F_C[:3, 3] = [.035, 0., .06]
    T_F_S = np.eye(4)
    T_F_S[:3, 3] = [0., 0., .16]
    T_B_C = T_B_F @ T_F_C
    p = transform_point(T_B_C, target['surface_point_color_m'])
    n = T_B_C[:3, :3] @ np.asarray(target['normal_out_color'])
    n /= np.linalg.norm(n)
    R_B_S = suction_rotation(n)
    # ONLY an illustration: R = Rz(rz) Ry(ry) Rx(rx), not a verified FR5 convention.
    euler = Rotation.from_matrix(R_B_S).as_euler('xyz', degrees=True)
    pre = p + .04 * n
    T_B_S = np.eye(4)
    T_B_S[:3, :3], T_B_S[:3, 3] = R_B_S, p
    T_B_F_target = T_B_S @ np.linalg.inv(T_F_S)
    return {'schema': 'blueberry_fr5_dry_run_v1', 'mode': 'simulation_only',
            'executable': False, 'source_frame': target['frame'], 'instance_id': target['instance_id'],
            'measured_inputs': {'point_color_m': target['surface_point_color_m'],
                                'normal_color': target['normal_out_color'], 'quality': target['quality']},
            'assumed_inputs': {'T_base_from_flange_at_capture': T_B_F.tolist(),
                               'T_flange_from_color_hand_eye': T_F_C.tolist(),
                               'T_flange_from_suction_tcp': T_F_S.tolist(),
                               'preapproach_distance_mm': 40,
                               'orientation_convention': 'illustration only: Rz(rz) @ Ry(ry) @ Rx(rx)',
                               'orientation_confirmed_for_fr5': False, 'tool_number': 1, 'user_number': 0},
            'derived': {'point_base_mm': (p * 1000).tolist(), 'normal_out_base': n.tolist(),
                        'R_base_from_suction': R_B_S.tolist(),
                        'preapproach_desc_pos_mm_deg': np.r_[pre * 1000, euler].tolist(),
                        'surface_desc_pos_mm_deg': np.r_[p * 1000, euler].tolist(),
                        'flange_target_for_reference_m': T_B_F_target.tolist()},
            'fr5_request_example': {'method': 'MoveL', 'desc_pos': np.r_[pre * 1000, euler].tolist(),
                                   'tool': 1, 'user': 0, 'vel': 5, 'vel_unit': 'percent_in_default_mode',
                                   'blendR': -1, 'offset_flag': 0},
            'not_validated': ['real_hand_eye', 'capture_time_robot_pose', 'real_tcp', 'fr5_euler_convention',
                              'maturity', 'cup_footprint', 'collision', 'inverse_kinematics',
                              'temporal_stability', 'local_surface_ground_truth'],
            'blocking_reason': 'Real robot calibration and capture pose are absent; never execute this payload.'}


def select_demo_target(targets):
    # Prefer an interior fruit at a useful observation distance. A noise floor
    # avoids rewarding a quantized, constant-depth patch as "perfect accuracy".
    preferred = [t for t in targets if
                 min(t['bbox_xyxy'][0], t['bbox_xyxy'][1], 640 - t['bbox_xyxy'][2], 480 - t['bbox_xyxy'][3]) > 8
                 and .35 <= t['surface_point_color_m'][2] <= .65]
    pool = preferred or targets
    return max(pool, key=lambda t: t['confidence'] * t['quality']['valid_depth_ratio'] /
               (1 + max(.5, t['quality'].get('plane_rms_mm', 999))))


def selected_visual(data, report, best):
    selected = cv2.imdecode(np.fromfile(str(data / 'rgb' / f'{best["frame"]}.png'), np.uint8), cv2.IMREAD_COLOR)
    selected_xyz = np.load(data / 'xyz_color' / f'{best["frame"]}.npz')['xyz_color_m']
    depth_vis = np.zeros(selected_xyz.shape[:2], np.uint8)
    finite = np.isfinite(selected_xyz[:, :, 2])
    depth_vis[finite] = np.clip((selected_xyz[:, :, 2][finite] - .2) / .8 * 255, 1, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_vis[~finite] = 0
    prediction = cv2.imdecode(np.fromfile(str(report / 'predictions' / f'{best["frame"]}.jpg'), np.uint8), cv2.IMREAD_COLOR)
    u, v = best['contact_rgb_pixel']
    for im in (selected, prediction, depth_vis):
        cv2.circle(im, (u, v), 22, (0, 255, 255), 2)
        cv2.drawMarker(im, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    contact_sheet([('Original RGB / selected fruit', selected), ('Segmentation + surface point', prediction),
                   ('Raw projected samples / no filling', depth_vis)], report / 'selected_pipeline.jpg', columns=3)
    b = best['bbox_xyxy']
    x0, y0, x1, y1 = max(0, int(b[0]) - 30), max(0, int(b[1]) - 30), min(640, int(b[2]) + 30), min(480, int(b[3]) + 30)
    crop = cv2.resize(selected[y0:y1, x0:x1], (500, 500), interpolation=cv2.INTER_NEAREST)
    imwrite(report / 'selected_fruit_crop.png', crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', default='data/20260902_121826.db3')
    parser.add_argument('--weights', default='runs/segment/gpu_public_640_50epochs/weights/best.pt')
    parser.add_argument('--data-output', default='data/processed/d435_20260902_geometry')
    parser.add_argument('--report-output', default='reports/localization_20261004_final')
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--max-delta-ms', type=float, default=12.)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--start-color-index', type=int, default=0)
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0 or args.max_delta_ms <= 0:
        parser.error('Sampling interval and synchronization threshold must be positive')
    data, report = project_path(args.data_output), project_path(args.report_output)
    for path in (data, report):
        if path.exists() and any(path.iterdir()):
            parser.error(f'Output is not empty: {path}; choose a new directory')
        path.mkdir(parents=True, exist_ok=True)
    for name in ('rgb', 'depth_z16', 'xyz_color', 'pseudo_labels'):
        (data / name).mkdir()
    for name in ('predictions', 'masks'):
        (report / name).mkdir()
    recording = Recording(project_path(args.recording))
    calibration = recording.calibration()
    sample_density = min(1., recording.depth_info['fx'] * recording.depth_info['fy'] /
                         (recording.color_info['fx'] * recording.color_info['fy']))
    write_json(data / 'calibration.json', calibration)
    write_json(report / 'calibration.json', calibration)
    pairs = list(recording.pairs(args.interval, args.max_delta_ms, args.start_color_index))
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    if not pairs:
        raise RuntimeError('No frame pairs passed the synchronization threshold')
    model = None
    weights = project_path(args.weights)
    if not args.extract_only:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        init_runtime(4)
        from ultralytics import YOLO
        model = YOLO(str(weights), task='segment')
    manifest, records, sheet, all_targets = [], [], [], []
    for i, pair in enumerate(pairs):
        frame = f'frame_{pair["color_index"]:06d}'
        rgb, rgb_header = recording.image(pair['color_id'])
        depth, depth_header = recording.image(pair['depth_id'])
        if rgb_header['stamp_ns'] != pair['color_timestamp_ns'] or depth_header['stamp_ns'] != pair['depth_timestamp_ns']:
            raise ValueError('Image-header and bag timestamps differ; synchronization needs a new policy')
        if rgb.shape != (calibration['rgb']['height'], calibration['rgb']['width'], 3):
            raise ValueError('RGB dimensions differ from recorded calibration')
        xyz = project_depth(depth, recording.depth_info, recording.color_info, recording.T_C_D, recording.scale)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        imwrite(data / 'rgb' / f'{frame}.png', bgr)
        imwrite(data / 'depth_z16' / f'{frame}.png', depth)
        np.savez_compressed(data / 'xyz_color' / f'{frame}.npz', xyz_color_m=xyz)
        meta_color = recording.metadata[recording.color_prefix].get(pair['color_timestamp_ns'], {})
        meta_depth = recording.metadata[recording.depth_prefix].get(pair['depth_timestamp_ns'], {})
        entry = pair | {'frame': frame, 'rgb_path': f'rgb/{frame}.png',
                        'depth_path': f'depth_z16/{frame}.png', 'xyz_path': f'xyz_color/{frame}.npz',
                        'group_id': recording.path.stem, 'split': 'unassigned',
                        'color_metadata': meta_color, 'depth_metadata': meta_depth,
                        'annotation_status': 'unlabeled' if model is None else 'pseudo_unreviewed'}
        manifest.append(entry)
        targets, label_rows = [], []
        overlay = bgr.copy()
        if model is not None:
            result = model.predict(bgr, device=device(args.device), imgsz=640, conf=args.conf,
                                   max_det=500, retina_masks=True, verbose=False)[0]
            overlay = result.plot()
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy() > .5
                if masks.shape[1:] != bgr.shape[:2]:
                    raise ValueError('Model masks are not in original RGB coordinates')
                np.savez_compressed(report / 'masks' / f'{frame}.npz', masks=masks)
                for k, mask in enumerate(masks):
                    cls = int(result.boxes.cls[k])
                    target = estimate_surface(mask, xyz, sample_density) | {
                        'frame': frame, 'instance_id': k, 'class_id': cls, 'class_name': result.names[cls],
                        'confidence': float(result.boxes.conf[k]),
                        'bbox_xyxy': result.boxes.xyxy[k].cpu().tolist(),
                        'coordinate_frame': 'color_optical', 'robot_executable': False,
                        'rgb_depth_delta_ms': pair['delta_ms']}
                    targets.append(target)
                    polygon = result.masks.xy[k] / np.array([rgb.shape[1], rgb.shape[0]])
                    label_rows.append(str(cls) + ' ' + ' '.join(f'{x:.7f}' for x in polygon.ravel()))
                    if target['status'] == 'candidate':
                        all_targets.append(target)
                        u, v = target['contact_rgb_pixel']
                        cv2.drawMarker(overlay, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                        cv2.putText(overlay, f'{k}: {target["surface_point_color_m"][2]*1000:.0f}mm',
                                    (u + 6, v), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
            (data / 'pseudo_labels' / f'{frame}.txt').write_text('\n'.join(label_rows), encoding='utf-8')
        imwrite(report / 'predictions' / f'{frame}.jpg', overlay)
        records.append({'frame': frame, 'color_timestamp_ns': pair['color_timestamp_ns'], 'targets': targets})
        if i % max(1, len(pairs) // 16) == 0 and len(sheet) < 16:
            sheet.append((f'{pair["color_timestamp_ns"]/1e9:.2f}s / {len(targets)} instances', overlay))
        print(f'{i+1}/{len(pairs)} {frame}: {len(targets)} detections, '
              f'{sum(t["status"] == "candidate" for t in targets)} geometry candidates', flush=True)
    write_json(data / 'manifest.json', manifest)
    (report / 'instances_3d.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False, allow_nan=False)
                                                        for r in records) + '\n', encoding='utf-8')
    contact_sheet(sheet, report / 'replay_contact_sheet.jpg')
    summary = {'recording': calibration['recording'], 'duration_s': (
        recording.frames[recording.color_prefix][-1][1] - recording.frames[recording.color_prefix][0][1]) / 1e9,
        'rgb_frames': len(recording.frames[recording.color_prefix]),
        'depth_frames': len(recording.frames[recording.depth_prefix]), 'processed_frames': len(pairs),
        'interval_s': args.interval, 'sync_threshold_ms': args.max_delta_ms,
        'pair_delta_ms': {'min': min(p['delta_ms'] for p in pairs), 'max': max(p['delta_ms'] for p in pairs),
                          'median': float(np.median([p['delta_ms'] for p in pairs]))},
        'detections': sum(len(r['targets']) for r in records), 'geometry_candidates': len(all_targets),
        'rejected': sum(t['status'] == 'reject' for r in records for t in r['targets']),
        'unique_fruit_count': None, 'note': 'Repeated frame detections are not unique fruit counts.',
        'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest() if model else None,
        'inference_confidence': args.conf, 'maturity_available': False,
        'source_database_opened_read_only': True, 'hardware_connected': False,
        'normal_note': 'PCA local normal is an unvalidated single-frame estimate.',
        'expected_projected_depth_sample_density': sample_density,
        'geometric_accuracy_verified': False}
    if all_targets:
        # Transparent heuristic chooses one demonstration, not the robot's pick order.
        best = select_demo_target(all_targets)
        summary['selected_target'] = best
        summary['demo_selection_note'] = 'Prefer fruit >8px from border at 0.35-0.65m; score uses 0.5mm residual floor.'
        write_json(report / 'selected_camera_target.json', best)
        write_json(report / 'fr5_simulation_only.json', simulate_robot(best))
        selected_visual(data, report, best)
    write_json(report / 'summary.json', summary)
    write_json(report / 'topic_inventory.json', recording.topics)
    recording.conn.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Read RealSense native ROS2 SQLite recordings and project raw depth.

This intentionally supports only the Image/String CDR messages and native
RealSense schema used by this recording, not arbitrary ROS bags. No ROS install
is needed. No robot connection or motion is provided.
"""
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np


class CDR:
    def __init__(self, blob):
        if blob[:4] != b'\x00\x01\x00\x00':
            raise ValueError('Only little-endian CDR v1 is supported')
        self.blob, self.pos = blob, 4

    def scalar(self, fmt, alignment=4):
        self.pos += (-(self.pos - 4)) % alignment
        value = struct.unpack_from('<' + fmt, self.blob, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return value

    def string(self):
        size = self.scalar('I')
        if size < 1 or self.pos + size > len(self.blob):
            raise ValueError('Invalid CDR string size')
        value = self.blob[self.pos:self.pos + size]
        self.pos += size
        if value[-1] != 0:
            raise ValueError('CDR string lacks terminator')
        return value[:-1].decode('utf-8')


def decode_string(blob):
    return CDR(blob).string()


def decode_image(blob):
    c = CDR(blob)
    sec, nsec, frame = c.scalar('i'), c.scalar('I'), c.string()
    height, width, encoding = c.scalar('I'), c.scalar('I'), c.string()
    bigendian, step, size = c.scalar('B', 1), c.scalar('I'), c.scalar('I')
    if size != height * step or c.pos + size != len(blob):
        raise ValueError('Image payload size/stride mismatch')
    if encoding == 'rgb8':
        if step < width * 3:
            raise ValueError('Invalid RGB stride')
        rows = np.frombuffer(blob, np.uint8, count=size, offset=c.pos).reshape(height, step)
        image = rows[:, :width * 3].reshape(height, width, 3).copy()
    elif encoding in ('mono16', '16UC1'):
        if step < width * 2 or step % 2:
            raise ValueError('Invalid depth stride')
        dtype = '>u2' if bigendian else '<u2'
        rows = np.frombuffer(blob, dtype, count=size // 2, offset=c.pos).reshape(height, step // 2)
        image = rows[:, :width].astype(np.uint16)
    else:
        raise ValueError(f'Unsupported image encoding {encoding}')
    return image, {'frame_id': frame, 'stamp_ns': sec * 10**9 + nsec,
                   'width': width, 'height': height, 'encoding': encoding, 'step': step}


def fields(text):
    return dict(item.split('=', 1) for item in text.strip(';').split(';') if '=' in item)


def intrinsics(text):
    values = fields(text)
    return {key: int(values[key]) for key in ('width', 'height')} | {
        key: float(values[key]) for key in ('fx', 'fy', 'ppx', 'ppy')} | {
        'model': values['model'], 'coeffs': list(map(float, values['coeffs'].split(',')))}


def native_extrinsics(text):
    values = fields(text)
    transform = np.eye(4)
    # rs2_extrinsics stores rotation COLUMN major. Native tf/ref_N is ref -> stream.
    transform[:3, :3] = np.array(list(map(float, values['rotation'].split(',')))).reshape(3, 3, order='F')
    transform[:3, 3] = list(map(float, values['translation'].split(',')))
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=5e-6):
        raise ValueError('Recorded rotation is not orthonormal within rounding tolerance')
    return transform


class Recording:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.conn = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
        self.topics = {name: {'id': tid, 'type': kind, 'serialization': serial}
                       for tid, name, kind, serial in self.conn.execute(
                           'select id,name,type,serialization_format from topics')}
        self.color_prefix = '/device_0/sensor_1/Color_0'
        self.depth_prefix = '/device_0/sensor_0/Depth_0'
        self.color_info = intrinsics(self.string(self.color_prefix + '/camera_info'))
        self.depth_info = intrinsics(self.string(self.depth_prefix + '/camera_info'))
        self.scale = float(self.string('/device_0/sensor_0/option/Depth_Units/value'))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid depth scale')
        tc_ref = native_extrinsics(self.string(self.color_prefix + '/tf/ref_0'))
        td_ref = native_extrinsics(self.string(self.depth_prefix + '/tf/ref_0'))
        self.T_C_D = tc_ref @ np.linalg.inv(td_ref)
        self.frames = {}
        for prefix in (self.color_prefix, self.depth_prefix):
            tid = self.topics[prefix + '/image/data']['id']
            self.frames[prefix] = self.conn.execute(
                'select id,timestamp from messages where topic_id=? order by timestamp,id', (tid,)).fetchall()
        # Meta is small; image blobs are fetched only by their primary keys.
        self.metadata = {}
        for prefix in (self.color_prefix, self.depth_prefix):
            tid = self.topics[prefix + '/image/metadata']['id']
            self.metadata[prefix] = {ts: fields(decode_string(blob)) for ts, blob in self.conn.execute(
                'select timestamp,data from messages where topic_id=?', (tid,))}

    def string(self, name):
        topic = self.topics[name]
        if topic['type'] != 'std_msgs/msg/String' or topic['serialization'] != 'cdr':
            raise ValueError(f'Unexpected native metadata schema: {name}')
        row = self.conn.execute('select data from messages where topic_id=? limit 1', (topic['id'],)).fetchone()
        if row is None:
            raise ValueError(f'Missing metadata {name}')
        return decode_string(row[0])

    def image(self, message_id):
        return decode_image(self.conn.execute('select data from messages where id=?', (message_id,)).fetchone()[0])

    def pairs(self, interval_s=1.0, max_delta_ms=12.0, start_color_index=0):
        colors = self.frames[self.color_prefix]
        depths = self.frames[self.depth_prefix]
        times = np.array([t for _, t in depths], dtype=np.int64)
        last = -10**18
        for ci, (color_id, ts) in enumerate(colors):
            if ci < start_color_index:
                continue
            if ts - last < interval_s * 1e9:
                continue
            j = int(np.searchsorted(times, ts))
            options = [i for i in (j - 1, j) if 0 <= i < len(times)]
            j = min(options, key=lambda i: abs(int(times[i]) - ts))
            delta = abs(int(times[j]) - ts) / 1e6
            if delta <= max_delta_ms:
                last = ts
                yield {'color_index': ci, 'color_id': color_id, 'depth_id': depths[j][0],
                       'color_timestamp_ns': ts, 'depth_timestamp_ns': int(times[j]),
                       'delta_ms': delta}

    def calibration(self):
        return {'recording': str(self.path), 'device': fields(self.string('/device_0/info')),
                'rgb': self.color_info, 'depth': self.depth_info, 'depth_scale_m': self.scale,
                'T_color_optical_from_depth_optical': self.T_C_D.tolist(),
                'extrinsics_storage': 'column_major; native tf/ref_0 maps ref to stream',
                'robot_pose_recorded': False, 'hand_eye_available': False,
                'metadata_precision_note': 'Native recorder serializes calibration to six decimal places.'}


def project_depth(depth, depth_info, color_info, T_C_D, scale, min_m=0.15, max_m=3.0):
    """Return RGB-sized XYZ map in COLOR optical coordinates, using nearest z buffer.

    No hole filling or synthetic depth. This recording has zero distortion;
    nonzero coefficients fail explicitly instead of silently using a pinhole.
    """
    for info in (depth_info, color_info):
        if any(abs(x) > 1e-12 for x in info['coeffs']):
            raise ValueError('Nonzero distortion requires a matching RealSense SDK projection implementation')
    if depth.shape != (depth_info['height'], depth_info['width']):
        raise ValueError('Depth shape differs from intrinsics')
    vv, uu = np.nonzero((depth * scale >= min_m) & (depth * scale <= max_m))
    z = depth[vv, uu].astype(float) * scale
    pd = np.column_stack(((uu - depth_info['ppx']) * z / depth_info['fx'],
                          (vv - depth_info['ppy']) * z / depth_info['fy'], z))
    pc = pd @ T_C_D[:3, :3].T + T_C_D[:3, 3]
    pc = pc[pc[:, 2] > 0]
    u = np.rint(pc[:, 0] / pc[:, 2] * color_info['fx'] + color_info['ppx']).astype(int)
    v = np.rint(pc[:, 1] / pc[:, 2] * color_info['fy'] + color_info['ppy']).astype(int)
    h, w = color_info['height'], color_info['width']
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pc, u, v = pc[valid], u[valid], v[valid]
    linear = v * w + u
    order = np.lexsort((pc[:, 2], linear))
    _, first = np.unique(linear[order], return_index=True)
    selected = order[first]
    xyz = np.full((h * w, 3), np.nan, np.float32)
    xyz[linear[selected]] = pc[selected]
    return xyz.reshape(h, w, 3)


def estimate_surface(mask, xyz, expected_sample_density=1.0):
    """Measure a conservative visible surface candidate; does not prove suction feasibility."""
    mask = mask.astype(np.uint8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    inner = distance >= 2.5
    finite = np.isfinite(xyz).all(axis=2)
    count = int(inner.sum())
    valid = inner & finite
    # Raw depth has a lower angular sample density than color. Empty projected
    # RGB pixels are often sampling gaps, not sensor failures. Normalize by the
    # approximate pinhole density, while retaining the raw occupancy for audit.
    if not 0 < expected_sample_density <= 1:
        raise ValueError('Expected projected sample density must lie in (0,1]')
    occupancy = float(valid.sum() / max(count, 1))
    quality = {'inner_pixels': count, 'valid_depth_pixels': int(valid.sum()),
               'projected_rgb_occupancy': round(occupancy, 4),
               'expected_projected_sample_density': expected_sample_density,
               'valid_depth_ratio': round(min(1., occupancy / expected_sample_density), 4),
               'coverage_note': 'Approximate sampling-density-normalized support, not measured sensor fill rate.'}
    reasons = []
    if count < 30 or valid.sum() < 20:
        return {'status': 'reject', 'reasons': ['too_few_interior_depth_points'], 'quality': quality}
    if quality['valid_depth_ratio'] < 0.6:
        reasons.append('low_valid_depth_ratio')
    values = xyz[:, :, 2][valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    cutoff = min(0.02, max(0.004, 3 * 1.4826 * mad))
    coherent = valid & (abs(xyz[:, :, 2] - median) <= cutoff)
    quality.update({'interior_depth_median_m': median, 'interior_depth_mad_mm': mad * 1000,
                    'depth_inlier_ratio': round(float(coherent.sum() / max(valid.sum(), 1)), 4)})
    if quality['depth_inlier_ratio'] < 0.8:
        reasons.append('mixed_depth_layers')
    if mad > 0.006:
        reasons.append('large_depth_spread')
    score = np.where(coherent, distance, -1)
    v, u = np.unravel_index(np.argmax(score), score.shape)
    yy, xx = np.indices(mask.shape)
    patch = coherent & ((xx - u)**2 + (yy - v)**2 <= 5**2)
    points = xyz[patch].astype(float)
    contact = xyz[v, u].astype(float)
    quality['local_point_count'] = len(points)
    normal = None
    if len(points) >= 12:
        center = np.mean(points, axis=0)
        eigenvalues, vectors = np.linalg.eigh(np.cov((points - center).T))
        normal = vectors[:, 0]
        if np.dot(normal, -contact) < 0:
            normal = -normal
        # Project the measured center pixel onto the fitted local surface.
        contact = contact - np.dot(contact - center, normal) * normal
        rms = float(np.sqrt(max(eigenvalues[0], 0))) * 1000
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(normal, -contact / np.linalg.norm(contact)), -1, 1))))
        quality.update({'plane_rms_mm': rms, 'normal_view_angle_deg': tilt,
                        'tangent_span_mm': float(np.sqrt(max(eigenvalues[1], 0))) * 1000,
                        'mask_edge_clearance_px': float(distance[v, u])})
        if rms > 2.0:
            reasons.append('noisy_local_surface')
        if tilt > 45:
            reasons.append('oblique_or_unreliable_normal')
        if quality['tangent_span_mm'] < 1:
            reasons.append('insufficient_normal_support')
    else:
        reasons.append('too_few_normal_points')
    return {'status': 'candidate' if not reasons else 'reject', 'reasons': reasons,
            'contact_rgb_pixel': [int(u), int(v)], 'surface_point_color_m': contact.tolist(),
            'normal_out_color': None if normal is None else normal.tolist(), 'quality': quality,
            'point_kind': 'visible_surface_candidate_not_fruit_center',
            'suction_feasibility_verified': False, 'maturity': 'unknown'}


def transform_point(T, p):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('Expected a 4x4 homogeneous transform')
    return T[:3, :3] @ np.asarray(p) + T[:3, 3]


def suction_rotation(normal_out, reference=(1., 0., 0.)):
    z = -np.asarray(normal_out, dtype=float)
    z /= np.linalg.norm(z)
    h = np.asarray(reference, dtype=float)
    if abs(np.dot(h / np.linalg.norm(h), z)) > 0.95:
        h = np.array([0., 1., 0.])
    x = h - np.dot(h, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

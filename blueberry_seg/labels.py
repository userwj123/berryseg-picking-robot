"""Recover approximate instance polygons from AgriVision foreground masks.

Binary masks do not encode true instance IDs. Both conversion modes produce
derived labels; watershed boundaries in particular require human review.
"""
import cv2
import numpy as np


def binary_mask(array):
    array = np.asarray(array)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise ValueError('Unsupported mask channel count')
        rgb = array[:, :, :3]
        if not np.all(rgb == rgb[:, :, :1]):
            raise ValueError('Color/instance-ID mask detected: do not use binary converter')
        array = rgb[:, :, 0]
    if array.ndim != 2 or not np.isin(array, [0, 1, 255]).all():
        raise ValueError('Expected a binary 0/1/255 foreground mask')
    return (array > 0).astype(np.uint8)


def instance_polygons(mask, method='components', min_area=30, peak_ratio=0.55):
    """Return polygons in pixel coordinates and auditable conversion counts."""
    if method not in ('components', 'watershed'):
        raise ValueError(f'Unknown instance method: {method}')
    mask = binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    polygons = []
    info = {'components': count-1, 'small_regions_removed': 0, 'split_components': 0,
            'derived_instances': 0, 'irregular_components': 0}
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            info['small_regions_removed'] += 1
            continue
        # Padding keeps distance transforms correct for components touching image borders.
        local = np.pad((labels[y:y+h, x:x+w] == label).astype(np.uint8), 1)
        regions = local.astype(np.int32)
        if method == 'watershed':
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
            radius = float(distance.max())
            peaks = peak_local_max(distance, min_distance=max(2, round(radius * peak_ratio)),
                                   threshold_rel=0.40, labels=local, exclude_border=False)
            if len(peaks) > 1:
                markers = np.zeros(local.shape, dtype=np.int32)
                markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks)+1)
                regions = watershed(-distance, markers, mask=local)
                info['split_components'] += 1
        for region in range(1, int(regions.max())+1):
            region_mask = (regions == region).astype(np.uint8)
            if int(region_mask.sum()) < min_area:
                info['small_regions_removed'] += 1
                continue
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            area_contour = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*np.pi*area_contour/(perimeter*perimeter) if perimeter else 0
            if circularity < 0.55:
                info['irregular_components'] += 1
            polygon = cv2.approxPolyDP(contour, max(0.35, 0.0015*perimeter), True).reshape(-1, 2)
            if len(polygon) < 3 or area_contour <= 0:
                info['small_regions_removed'] += 1
                continue
            polygon = polygon.astype(np.float64) + [int(x)-1, int(y)-1]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, mask.shape[1]-1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, mask.shape[0]-1)
            polygons.append(polygon)
    info['derived_instances'] = len(polygons)
    return polygons, info


def yolo_rows(polygons, width, height):
    rows = []
    for polygon in polygons:
        coords = polygon / np.array([width, height], dtype=float)
        rows.append('0 ' + ' '.join(f'{v:.7f}' for v in coords.reshape(-1)))
    return rows


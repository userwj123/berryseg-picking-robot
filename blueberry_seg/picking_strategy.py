"""Blueberry picking strategy: distance filter -> ripeness filter -> TSP-based picking order -> 6DOF output.

Picking order rationale (from literature):
- Traveling Salesman Problem (TSP) is the standard formulation for continuous fruit picking:
  minimize total end-effector travel distance while visiting all targets once.
  References:
    * Peiper & Sarig (1991), Near-Minimum-Time Task Planning for Fruit-Picking Robots
    * Kiani et al. (2023), Three-dimensional continuous picking path planning based on OSACO
    * Zhang et al. (2025), MSI-RRTCHP for continuous cherry picking (improved TSP + RRT)
    * HAL (2025), Two Arms One Goal: TSP-based scheduling for dual-arm harvesting
- For small N (< 30), exact TSP solvers (dynamic programming, branch and bound) are feasible.
  For larger N, heuristic TSP (nearest neighbor + 2-opt) is used.
- Additional practical rules from literature:
    * Pick from periphery to center to avoid collisions (tomato harvesting, Yu et al. 2022)
    * Prefer fewer neighbors first (reduce target displacement)
    * Distance filter before ripeness filter (meeting minutes, 2026-10-08)

Output per pick: 6 values for robot arm:
    [X, Y, Z, normal_X, normal_Y, normal_Z] in camera/optical frame (meters, unitless normal).
"""
import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  Data structures                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class BerryCandidate:
    """One detected blueberry with 3D position and ripeness."""
    frame: str
    instance_id: int
    confidence: float
    bbox_xyxy: List[float]
    polygon_xy: List[List[float]]
    surface_point_color_m: Tuple[float, float, float]  # X, Y, Z in camera frame (m)
    normal_out_color: Tuple[float, float, float]       # unit normal (unitless)
    quality: dict
    ripeness: str = "unknown"  # "ripe", "unripe", "semi_ripe", "unknown"
    distance_m: float = field(init=False)

    def __post_init__(self):
        self.distance_m = math.sqrt(
            self.surface_point_color_m[0] ** 2 +
            self.surface_point_color_m[1] ** 2 +
            self.surface_point_color_m[2] ** 2
        )

    def to_robot_6dof(self) -> List[float]:
        """Return [X, Y, Z, nx, ny, nz] for robot arm."""
        return [
            round(self.surface_point_color_m[0], 6),
            round(self.surface_point_color_m[1], 6),
            round(self.surface_point_color_m[2], 6),
            round(self.normal_out_color[0], 6),
            round(self.normal_out_color[1], 6),
            round(self.normal_out_color[2], 6),
        ]


@dataclass
class PickingConfig:
    max_distance_m: float = 1.0          # step 1: ignore > 1m
    min_confidence: float = 0.1
    ripeness_allowed: Tuple[str, ...] = ("ripe",)   # step 2: only ripe proceeds
    # TSP settings
    tsp_method: str = "auto"             # "exact", "nn_2opt", "auto"
    exact_tsp_max_n: int = 22            # exact DP is O(n^2 * 2^n), keep small
    # practical heuristics from literature
    prefer_periphery: bool = True        # pick fewer neighbors first (tomato literature)
    neighbor_radius_m: float = 0.08      # radius to count neighbors (berry size + gripper margin)


# --------------------------------------------------------------------------- #
#  TSP solvers                                                                #
# --------------------------------------------------------------------------- #
def _euclidean(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def tsp_exact(points: List[Tuple[float, float, float]]) -> List[int]:
    """Exact TSP via Held-Karp DP. Returns visit order indices (0 = start)."""
    n = len(points)
    if n <= 1:
        return list(range(n))
    # cost matrix
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = _euclidean(points[i], points[j])
    # dp[mask][i] = min cost to visit mask ending at i
    INF = float('inf')
    dp = [[INF]*n for _ in range(1 << n)]
    parent = [[-1]*n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for mask in range(1 << n):
        for i in range(n):
            if not (mask & (1 << i)) or dp[mask][i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                nmask = mask | (1 << j)
                cost = dp[mask][i] + dist[i][j]
                if cost < dp[nmask][j]:
                    dp[nmask][j] = cost
                    parent[nmask][j] = i
    # find best end
    full = (1 << n) - 1
    best_end = min(range(n), key=lambda i: dp[full][i])
    # reconstruct
    order = []
    mask, i = full, best_end
    while i != -1:
        order.append(i)
        pi = parent[mask][i]
        mask ^= (1 << i)
        i = pi
    return order[::-1]


def tsp_nn_2opt(points: List[Tuple[float, float, float]]) -> List[int]:
    """Nearest neighbor + 2-opt improvement. Returns visit order indices."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    # nearest neighbor
    unvisited = set(range(1, n))
    order = [0]
    while unvisited:
        last = order[-1]
        nxt = min(unvisited, key=lambda j: _euclidean(points[last], points[j]))
        order.append(nxt)
        unvisited.remove(nxt)
    # 2-opt
    def tour_length(o):
        return sum(_euclidean(points[o[i]], points[o[(i+1) % n]]) for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                if tour_length(new_order) < tour_length(order) - 1e-9:
                    order = new_order
                    improved = True
    return order


def solve_tsp(points: List[Tuple[float, float, float]], method: str = "auto") -> List[int]:
    """Return visit order (indices into points)."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if method == "auto":
        method = "exact" if n <= 22 else "nn_2opt"
    if method == "exact":
        return tsp_exact(points)
    return tsp_nn_2opt(points)


# --------------------------------------------------------------------------- #
#  Picking strategy                                                           #
# --------------------------------------------------------------------------- #
class PickingStrategy:
    def __init__(self, config: Optional[PickingConfig] = None):
        self.config = config or PickingConfig()
        self.picked_history: List[BerryCandidate] = []

    # -------------------------- filters ------------------------------------ #
    def filter_distance(self, berries: List[BerryCandidate]) -> List[BerryCandidate]:
        """Step 1: ignore berries beyond max_distance_m."""
        return [b for b in berries if b.distance_m <= self.config.max_distance_m]

    def filter_ripeness(self, berries: List[BerryCandidate]) -> List[BerryCandidate]:
        """Step 2: keep only allowed ripeness levels."""
        return [b for b in berries if b.ripeness in self.config.ripeness_allowed]

    def filter_confidence(self, berries: List[BerryCandidate]) -> List[BerryCandidate]:
        return [b for b in berries if b.confidence >= self.config.min_confidence]

    def filter_all(self, berries: List[BerryCandidate]) -> List[BerryCandidate]:
        return self.filter_confidence(self.filter_ripeness(self.filter_distance(berries)))

    # -------------------------- ordering ----------------------------------- #
    def _neighbor_count(self, berry: BerryCandidate, all_berries: List[BerryCandidate]) -> int:
        """Count how many other berries are within neighbor_radius_m."""
        cnt = 0
        for b in all_berries:
            if b.frame == berry.frame and b.instance_id == berry.instance_id:
                continue
            if _euclidean(berry.surface_point_color_m, b.surface_point_color_m) <= self.config.neighbor_radius_m:
                cnt += 1
        return cnt

    def order_targets(self, berries: List[BerryCandidate]) -> List[BerryCandidate]:
        """Order berries for picking: TSP + optional periphery-first heuristic."""
        if not berries:
            return []
        # TSP on 3D points (start from first berry as anchor)
        points = [b.surface_point_color_m for b in berries]
        order_idx = solve_tsp(points, self.config.tsp_method)
        ordered = [berries[i] for i in order_idx]
        # optional: within each local cluster, pick fewer neighbors first
        if self.config.prefer_periphery:
            ordered = sorted(ordered,
                             key=lambda b: self._neighbor_count(b, berries))
        return ordered

    # -------------------------- output ------------------------------------- #
    def next_pick(self, berries: List[BerryCandidate]) -> Optional[BerryCandidate]:
        """Return the single next berry to pick (first in ordered list)."""
        ordered = self.order_targets(self.filter_all(berries))
        return ordered[0] if ordered else None

    def pick(self, berries: List[BerryCandidate]) -> Tuple[Optional[BerryCandidate], List[BerryCandidate]]:
        """Pick next berry, return (picked, remaining)."""
        picked = self.next_pick(berries)
        if picked is None:
            return None, []
        remaining = [b for b in berries
                     if not (b.frame == picked.frame and b.instance_id == picked.instance_id)]
        self.picked_history.append(picked)
        return picked, remaining

    def generate_robot_commands(self, berries: List[BerryCandidate], max_picks: int = 6) -> List[dict]:
        """Generate up to max_picks robot commands in TSP order."""
        queue = list(berries)
        commands = []
        for _ in range(max_picks):
            picked, queue = self.pick(queue)
            if picked is None:
                break
            commands.append({
                'frame': picked.frame,
                'instance_id': picked.instance_id,
                'distance_m': round(picked.distance_m, 3),
                'confidence': round(picked.confidence, 3),
                'ripeness': picked.ripeness,
                'robot_6dof': picked.to_robot_6dof(),
            })
        return commands

    def reset(self):
        self.picked_history.clear()


# --------------------------------------------------------------------------- #
#  I/O helpers                                                                #
# --------------------------------------------------------------------------- #
def load_berries_from_jsonl(path: str) -> List[BerryCandidate]:
    """Load detections from localize_recording.py instances_3d.jsonl format."""
    berries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = record['frame']
            for t in record['targets']:
                if t['status'] != 'candidate':
                    continue
                b = BerryCandidate(
                    frame=frame,
                    instance_id=t['instance_id'],
                    confidence=t['confidence'],
                    bbox_xyxy=t['bbox_xyxy'],
                    polygon_xy=[],
                    surface_point_color_m=tuple(t['surface_point_color_m']),
                    normal_out_color=tuple(t['normal_out_color']),
                    quality=t.get('quality', {}),
                    ripeness=t.get('maturity', 'unknown'),
                )
                berries.append(b)
    return berries


# --------------------------------------------------------------------------- #
#  Demo / test                                                                #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    demo = [
        BerryCandidate('f1', 1, 0.9, [100, 100, 200, 200], [], (0.1, 0.0, 0.4), (0, 0, -1), {}, 'ripe'),
        BerryCandidate('f1', 2, 0.8, [300, 100, 400, 200], [], (0.2, 0.0, 0.6), (0, 0, -1), {}, 'unripe'),
        BerryCandidate('f1', 3, 0.7, [500, 100, 600, 200], [], (0.0, 0.1, 0.3), (0, 0, -1), {}, 'ripe'),
        BerryCandidate('f1', 4, 0.6, [700, 100, 800, 200], [], (0.0, 0.0, 0.7), (0, 0, -1), {}, 'semi_ripe'),
        BerryCandidate('f1', 5, 0.5, [900, 100, 1000, 200], [], (0.0, 0.0, 0.45), (0, 0, -1), {}, 'unknown'),
        BerryCandidate('f1', 6, 0.95, [200, 200, 300, 300], [], (0.05, 0.05, 0.35), (0, 0, -1), {}, 'ripe'),
    ]
    s = PickingStrategy(PickingConfig(max_distance_m=1.0, min_confidence=0.2, ripeness_allowed=('ripe',)))
    print("Picking commands (max 6):")
    for cmd in s.generate_robot_commands(demo, max_picks=6):
        print(cmd)

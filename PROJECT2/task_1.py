# task_1.py (치명적 수선면 폭 계산 버그가 수정된 물리 엔진)
import numpy as np
from shapely.geometry import Polygon, box, LineString
from scipy.optimize import brentq
from shapely.affinity import rotate

def create_shape(shape_type, B, H):
    if shape_type == 'Rectangle':
        return Polygon([(-B/2, 0), (B/2, 0), (B/2, H), (-B/2, H)])
    elif shape_type == 'Triangle':
        return Polygon([(0, 0), (B/2, H), (-B/2, H)])
    elif shape_type == 'Semicircle':
        R = B / 2
        angles = np.linspace(np.pi, 2*np.pi, 50)
        pts = [(R*np.cos(a), R*np.sin(a) + R) for a in angles]
        pts.extend([(-R, R), (R, R)])
        return Polygon(pts).convex_hull

def calculate_stability(poly, target_area):
    minx, miny, maxx, maxy = poly.bounds
    
    def area_error(y):
        water = box(-100, -100, 100, y)
        return poly.intersection(water).area - target_area
        
    try:
        water_level = brentq(area_error, miny - 0.1, maxy + 0.1)
    except (ValueError, ZeroDivisionError):
        return None, None, None
        
    water_poly = box(-100, -100, 100, water_level)
    submerged = poly.intersection(water_poly)
    CB = submerged.centroid.coords[0]
    
    wl_line = LineString([(-100, water_level), (100, water_level)])
    
    # [버그 수정 완료] exterior를 제거하여 선분이 정상적으로 잡히게 합니다.
    intersection = poly.intersection(wl_line)
    
    W = intersection.length if hasattr(intersection, 'length') else 0
    I = (W**3) / 12
    BM = I / target_area if target_area > 0 else 0
    
    return CB, BM, water_level
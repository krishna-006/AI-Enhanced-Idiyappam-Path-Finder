# idiyappam_untangle.py
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label
import math
import uuid

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def _get_neighbors(pt, h, w):
    x, y = pt
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w:
                yield (nx, ny)

def _order_path(component_mask):
    ys, xs = np.where(component_mask)
    coords = list(zip(xs.tolist(), ys.tolist()))
    if not coords:
        return []
    coord_set = set(coords)
    h, w = component_mask.shape
    neighbors = {}
    for (x,y) in coords:
        nb = []
        for nx, ny in _get_neighbors((x,y), h, w):
            if (nx, ny) in coord_set:
                nb.append((nx, ny))
        neighbors[(x,y)] = nb
    endpoints = [p for p, nb in neighbors.items() if len(nb) == 1]
    if endpoints:
        start = endpoints[0]
    else:
        start = coords[0]
    visited = set([start])
    path = [start]
    curr = start
    while True:
        unvisited_nb = [p for p in neighbors[curr] if p not in visited]
        if unvisited_nb:
            nxt = unvisited_nb[0]
            path.append(nxt)
            visited.add(nxt)
            curr = nxt
            continue
        moved = False
        for idx in range(len(path)-2, -1, -1):
            node = path[idx]
            unvisited_nb = [p for p in neighbors[node] if p not in visited]
            if unvisited_nb:
                nxt = unvisited_nb[0]
                path = path[:idx+1]
                path.append(nxt)
                visited.add(nxt)
                curr = nxt
                moved = True
                break
        if not moved:
            break
    remaining = coord_set - visited
    while remaining:
        last = path[-1]
        nxt = min(remaining, key=lambda p: (p[0]-last[0])**2 + (p[1]-last[1])**2)
        stack = [nxt]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            path.append(node)
            for nb in neighbors[node]:
                if nb not in visited:
                    stack.append(nb)
        remaining = coord_set - visited
    return path

def _path_length(path):
    if len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(path)):
        x1,y1 = path[i-1]
        x2,y2 = path[i]
        length += math.hypot(x2-x1, y2-y1)
    return length

def process_idiyappam(input_path, out_folder, pixels_per_cm=None):
    """
    Returns:
      out_name (filename saved under out_folder),
      lengths_px (list of lengths in pixels),
      lengths_cm (list or None),
      total_len_px (float)
    """
    ensure_folder(out_folder)
    img_color = cv2.imread(input_path)
    if img_color is None:
        raise FileNotFoundError(f"Could not read {input_path}")

    # resize for performance
    max_side = 1200
    h0,w0 = img_color.shape[:2]
    if max(h0,w0) > max_side:
        scale = max_side / max(h0,w0)
        img_color = cv2.resize(img_color, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure noodles are white foreground
    if np.mean(bw==255) < 0.5:
        bw = cv2.bitwise_not(bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    skel = skeletonize(bw > 0).astype(np.uint8)
    labeled = label(skel, connectivity=2)
    num = labeled.max()

    lengths_px = []
    overlay = img_color.copy()

    for comp_id in range(1, num+1):
        component_mask = (labeled == comp_id)
        if component_mask.sum() < 8:
            continue
        path = _order_path(component_mask)
        if not path:
            continue
        length_px = _path_length(path)
        lengths_px.append(length_px)
        color = tuple(int(x) for x in np.random.randint(80, 240, size=3).tolist())
        for i in range(1, len(path)):
            x1,y1 = path[i-1]; x2,y2 = path[i]
            cv2.line(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color, 1)

    total_len_px = float(sum(lengths_px))
    lengths_cm = None
    if pixels_per_cm and pixels_per_cm > 0:
        lengths_cm = [l / pixels_per_cm for l in lengths_px]

    uid = uuid.uuid4().hex[:8]
    out_name = f"processed_{uid}.png"
    out_path = os.path.join(out_folder, out_name)
    cv2.imwrite(out_path, overlay)

    return out_name, lengths_px, lengths_cm, total_len_px

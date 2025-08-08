import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import networkx as nx

# --------------------
# Load and Preprocess Image
# --------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img_normalized = img / 255.0
    return img, img_normalized.reshape(1, 256, 256, 1)

# --------------------
# Draw Start and End Points
# --------------------
def draw_points(img, points):
    x1, y1, x2, y2 = [int(p) for p in points]
    cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)  # Start - Green
    cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1)  # End - Red
    return img

# --------------------
# Skeletonization
# --------------------
def get_skeleton(img):
    from skimage.morphology import skeletonize
    img = (img > 127).astype(np.uint8)
    skeleton = skeletonize(img).astype(np.uint8)
    return skeleton

# --------------------
# Detect Tangle Zones
# --------------------
def get_tangle_map(skeleton):
    kernel = np.array([[1,1,1], [1,10,1], [1,1,1]])
    neighbors = convolve(skeleton, kernel, mode='constant', cval=0)
    tangle_map = (neighbors >= 13).astype(np.uint8)
    tangle_score = np.sum(tangle_map)
    print("Tangle Score:", tangle_score)
    return tangle_map

# --------------------
# Convert Skeleton to Graph
# --------------------
def skeleton_to_graph(skeleton, tangle_map=None):
    G = nx.Graph()
    h, w = skeleton.shape
    for y in range(h):
        for x in range(w):
            if skeleton[y, x]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx_ = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_]:
                            weight = 1
                            if tangle_map is not None and tangle_map[ny, nx_] == 1:
                                weight = 1000
                            G.add_edge((x, y), (nx_, ny), weight=weight)
    return G

# --------------------
# Find Nearest White Pixel
# --------------------
def find_nearest_white(skeleton, x, y):
    h, w = skeleton.shape
    min_dist = float('inf')
    nearest = (x, y)
    for yy in range(h):
        for xx in range(w):
            if skeleton[yy, xx]:
                dist = (x - xx)**2 + (y - yy)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest = (xx, yy)
    return nearest

# --------------------
# Find All Endpoints
# --------------------
def find_all_endpoints(skeleton):
    endpoints = []
    h, w = skeleton.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x]:
                neighborhood = skeleton[y-1:y+2, x-1:x+2]
                if np.sum(neighborhood) == 2:
                    endpoints.append((x, y))
    return endpoints

# --------------------
# Untangle Path to Straight Line
# --------------------
def untangle_path(path, img):
    unrolled = np.zeros((20, len(path), 3), dtype=np.uint8)
    for i, (x, y) in enumerate(path):
        color = img[y, x] if 0 <= y < img.shape[0] and 0 <= x < img.shape[1] else 0
        unrolled[:, i] = color
    return unrolled

# --------------------
# Main Inference Function for All Noodles
# --------------------
def untangle_all(image_path):
    original_img, _ = preprocess_image(image_path)
    output_img = cv2.cvtColor(cv2.resize(original_img, (256, 256)), cv2.COLOR_GRAY2BGR)

    skeleton = get_skeleton(original_img)
    tangle_map = get_tangle_map(skeleton)
    G = skeleton_to_graph(skeleton, tangle_map)

    endpoints = find_all_endpoints(skeleton)
    used_pixels = set()

    count = 0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            start = endpoints[i]
            end = endpoints[j]
            if start in used_pixels or end in used_pixels:
                continue
            try:
                path = nx.shortest_path(G, source=start, target=end, weight='weight')
                if len(path) > 30:  # Skip very short paths
                    unrolled = untangle_path(path, output_img)
                    count += 1
                    plt.subplot(1, 2, 1)
                    for (x, y) in path:
                        cv2.circle(output_img, (x, y), 1, (0, 255, 255), -1)
                    plt.imshow(output_img)
                    plt.title(f"Path {count}")
                    plt.axis('off')

                    plt.subplot(1, 2, 2)
                    plt.imshow(unrolled)
                    plt.title("Untangled")
                    plt.axis('off')
                    plt.show()

                    used_pixels.update(path)
            except:
                continue

# --------------------
# Run the Script
# --------------------
untangle_all("C:/Users/Krishna/Documents/idiyappam_samplee.jpg")
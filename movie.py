import cv2
import numpy as np
from pathlib import Path
import time
import math
from tqdm import tqdm

#Opinion is 3138
# IMG_DIR = './training_images/health/'
IMG_DIR = './archive/natural_images/motorbike/'
WINDOW_WIDTH = 800
FPS = 30

RADIUS = 300
BOX_SIZE = 400

def play_images(folder:str, fps:int=24) -> None:
    """
    Plays all images in a folder (including subdirectories) at the specified fps
    """
    img_paths = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])

    num_imgs = len(img_paths)
    print(f'{num_imgs} images, estimated duration of {(num_imgs/fps):.2f}s')

    start_time = time.time()

    #ordering = sort_box_average(img_paths, BOX_SIZE)
    #ordering = sort_radius_average(img_paths, RADIUS)
    ordering = sort_by_histogram_tsp(img_paths)

    sorted_paths = [img_path for _, img_path in ordering]

    cv2.namedWindow('Image Sequence', cv2.WINDOW_KEEPRATIO)

    t2 = time.time()
    for i in range(len(sorted_paths)):
        t1 = time.time()

        img = cv2.imread(str(sorted_paths[i]))
        if img is None:
            continue

        # if i < num_imgs - 1:
        #     next_img = cv2.imread(str(sorted_paths[i+1]))
        #     inter_img = interpolate_linear(img, next_img, alpha=0.5)
        #     inter_img = resize_image(inter_img, WINDOW_WIDTH)

        img = resize_image(img, WINDOW_WIDTH)

        cv2.imshow('Image Sequence', img)

        wait_time = math.floor(max(0, (1000/fps) - (t2 - t1)))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            print('Quitting')
            break

        # if inter_img is not None:
        #     cv2.imshow('Image Sequence', inter_img)
        #     if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        #         print('Quitting')
        #         break

        t2 = time.time()

    cv2.destroyAllWindows()
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')


def resize_image(img: np.ndarray, target_width:int) -> None:
    (h, w) = img.shape[:2]
    scale_factor_w = target_width / float(w)
    new_width = target_width
    new_height = int(h * scale_factor_w)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def interpolate_linear(img1:np.ndarray, img2:np.ndarray, alpha:float) -> np.ndarray:
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

def sort_box_average(img_paths:list, box_size:int) -> list:
    ordering = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        x1, y1 = max(cx - box_size, 0), max(cy - box_size, 0)
        x2, y2 = min(cx + box_size, w), min(cy + box_size, h)
        center_crop = img[y1:y2, x1:x2]
        avg_color = np.mean(center_crop, axis=(0, 1))
        avg_sum = np.sum(avg_color)
        ordering.append((avg_sum, img_path))
    return ordering

def sort_radius_average(img_paths:list, radius:int) -> list:
    ordering = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = dist_from_center <= radius
        masked_pixels = img[mask]
        avg_color = np.mean(masked_pixels, axis=0)
        avg_sum = np.sum(avg_color)
        ordering.append((avg_sum, img_path))
    return ordering

def sort_by_histogram_tsp(img_paths:list, bins:tuple=(16, 16, 16)) -> list:
    """
    Sort images using greedy TSP based on color histogram similarity.
    """
    histograms = []
    valid_paths = []

    # Compute histograms
    for img_path in tqdm(img_paths, desc='Computing histograms'):
        img = cv2.imread(str(img_path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
        valid_paths.append(img_path)

    # Compute pairwise distance matrix
    N = len(histograms)
    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc='Computing distance matrix'):
        for j in range(i + 1, N):
            d = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CHISQR)
            dist_matrix[i, j] = dist_matrix[j, i] = d
        dist_matrix[i, i] = np.inf

    # Greedy TSP traversal
    visited = [False] * N
    path = [0]
    visited[0] = True
    for _ in tqdm(range(N - 1), desc='Finding TSP path'):
        last = path[-1]
        next_idx = np.argmin([dist_matrix[last][j] if not visited[j] else np.inf for j in range(N)])
        path.append(next_idx)
        visited[next_idx] = True

    return [(i, valid_paths[idx]) for i, idx in enumerate(path)]


if __name__ == '__main__':
    play_images(IMG_DIR, fps=FPS)

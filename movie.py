import cv2
import numpy as np
from pathlib import Path
import time
import math

WINDOW_WIDTH = 800

RADIUS = 300
BOX_SIZE = 400

def play_images_by_avg_color(folder:str, fps:int=24) -> None:
    """
    Plays all images in a folder (including subdirectories) at the specified fps,
    ordered by the average color of the center 100x100 pixels of each image.
    """
    img_paths = sorted([p for p in Path(folder).rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])

    num_imgs = len(img_paths)
    print(f'{num_imgs} images, estimated duration of {(num_imgs/fps):.2f}s')

    start_time = time.time()

    #ordering = alg_box_average(img_paths, BOX_SIZE)
    ordering = alg_radius_average(img_paths, RADIUS)

    ordering.sort()
    sorted_paths = [img_path for _, img_path in ordering]

    cv2.namedWindow('Image Sequence', cv2.WINDOW_KEEPRATIO)

    for img_path in sorted_paths:
        t1 = time.time()

        img = cv2.imread(str(img_path))
        img = resize_image(img, WINDOW_WIDTH)
        if img is None:
            continue

        # Show circle for radius
        #h, w = img.shape[:2]
        #cx, cy = w // 2, h // 2
        #cv2.circle(img, (cx, cy), RADIUS, (0, 0, 255), 2)

        cv2.imshow('Image Sequence', img)
        t2 = time.time()
        wait_time = math.floor(max(0, (1000/fps) - (t2 - t1)))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            print('Quitting')
            break

    cv2.destroyAllWindows()
    end_time = time.time()
    print(f'Duration: {end_time - start_time}')

def resize_image(img: np.ndarray, target_width:int) -> None:
    (h, w) = img.shape[:2]
    scale_factor_w = target_width / float(w)
    new_width = target_width
    new_height = int(h * scale_factor_w)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

def alg_box_average(img_paths:list, box_size:int) -> list:
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

def alg_radius_average(img_paths:list, radius:int) -> list:
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

if __name__ == '__main__':
    play_images_by_avg_color('./training_images/health/', fps=60)

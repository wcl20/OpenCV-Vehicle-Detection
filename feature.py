import cv2
import numpy as np

def hog(image, visualize=False):

    height, width, _ = image.shape

    # Global Normalization
    image = np.sqrt(image)
    image = np.rint(image).astype(np.uint8)

    # Calculate gradient
    gxs = np.zeros_like(image, dtype=np.double)
    gys = np.zeros_like(image, dtype=np.double)
    for channel in range(image.shape[2]):
        gxs[:,:,channel] = cv2.Sobel(image[:,:,channel], cv2.CV_32F, 1, 0, ksize=3)
        gys[:,:,channel] = cv2.Sobel(image[:,:,channel], cv2.CV_32F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitudes = np.hypot(gxs, gys)
    # Select gradient with largest magnitude in each channel
    rr, cc = np.meshgrid(np.arange(height), np.arange(width), indexing='ij', sparse=True)
    gx = gxs[rr, cc, magnitudes.argmax(axis=2)]
    gy = gys[rr, cc, magnitudes.argmax(axis=2)]
    magnitudes = np.hypot(gx, gy)

    # Calculate gradient orientation
    orientations = np.arctan2(gy, gx)
    orientations = np.rad2deg(orientations) % 180

    # Calculate histogram for 8x8 cells (9 orientations)
    histograms = np.zeros((height // 8, width // 8, 9))
    for h in range(0, height - 7, 8):
        for w in range(0, width - 7, 8):
            rr, cc = np.meshgrid(np.arange(h, h + 8), np.arange(w, w + 8), indexing='ij', sparse=True)
            orientation = orientations[h:h+8, w:w+8].ravel()
            magnitude = magnitudes[h:h+8, w:w+8].ravel()
            histogram, _ = np.histogram(orientation, bins=np.arange(0, 200, 20), weights=magnitude)
            histograms[h//8, w//8, :] = histogram

    # Normalize histograms for 16x16 blocks
    blocks = np.zeros((height // 8 - 1, width // 8 - 1, 9 * 4))
    for h in range(height // 8 - 1):
        for w in range(width // 8 - 1):
            block = histograms[h:h+2, w:w+2].ravel()
            blocks[h, w, :] = block / (np.linalg.norm(block) + 1e-5)

    # Visualize descriptors
    if visualize:
        for h in range(histograms.shape[0]):
            for w in range(histograms.shape[1]):
                center = (8 * w + 4, 8 * h + 4)
                angle = np.pi * (histograms.argmax(axis=2)[h][w] + 0.5) / 9
                if histograms.max(axis=2)[h][w] > 500:
                    dx = np.cos(angle)
                    dy = np.sin(angle)
                    pt1 = (int(8 * w + 4 - dx), int(8 * h + 4 - dy))
                    pt2 = (int(8 * w + 4 + dx), int(8 * h + 4 + dy))
                    cv2.line(image, pt1, pt2, (0, 0, 255), 1)
        cv2.imshow("HOG", image)
        cv2.waitKey(0)

    # return descriptor
    return blocks

def get_features(image):
    # Convert color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Resize image
    height, width, _ = image.shape
    scale = 64 / height
    resized = cv2.resize(hls, (int(width * scale), 64), interpolation=cv2.INTER_AREA)
    # Extract HOG features
    return hog(resized)

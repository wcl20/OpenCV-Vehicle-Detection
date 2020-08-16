import cv2
from feature import get_features

class Slider:

    def __init__(self, model):
        self.model = model

    def get_bounding_boxes(self, image):
        height, width, _ = image.shape
        bounding_boxes = []
        # Slider Window sizes
        for size in [100, 120, 160]:
            for y in [int(height * 0.5),int(height * 0.55), int(height * 0.6)]:
                # Get Region of interest (Horizon of driver's view)
                strip = image[y:y+size]
                # Get descriptor of strip
                scale = 64 / size
                descriptor = get_features(strip)
                for x in range(0, width - size + 1, size // 5):
                    # Extract feature from window
                    x_start = int(x * scale) // 8
                    features = descriptor[:, x_start:x_start+7]
                    if self.model.predict(features.ravel()):
                        bounding_boxes.append((x, y, x + size, y + size))
        return bounding_boxes

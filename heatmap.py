import cv2
import numpy as np

class Heatmap:

    def __init__(self, width, height):
        self.map = np.zeros((height, width)).astype(np.uint8)
        # Store bounding boxes
        self.memory = []

    def update(self, bounding_boxes):

        # Check memory exceed limit
        if len(self.memory) > 40:
            # Update heatmap
            for x1, y1, x2, y2 in self.memory.pop(0):
                self.map[y1:y2, x1:x2] -= 1

        # Save to memory
        self.memory.append(bounding_boxes)
        # Update heatmap
        for x1, y1, x2, y2 in bounding_boxes:
            self.map[y1:y2, x1:x2] += 1

    def get_bounding_boxes(self):
        # Apply threshold
        _, thresh = cv2.threshold(self.map, 30, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Return bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                yield (x, y, x + w, y + h)

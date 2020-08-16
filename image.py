import cv2
from model import Model, get_model
from slider import Slider
from heatmap import Heatmap

def main():
    # Read image
    image = cv2.imread("image.jpg")
    height, width, _ = image.shape
    # Create SVM Model
    model = get_model("model.pickle")
    # Create Heatmap
    heatmap = Heatmap(width, height)
    # Create slider
    slider = Slider(model)
    bounding_boxes = slider.get_bounding_boxes(image)
    for x1, y1, x2, y2 in bounding_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    # Update heatmap
    heatmap.update(bounding_boxes)
    for x1, y1, x2, y2 in heatmap.get_bounding_boxes():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

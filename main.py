import cv2
from model import Model, get_model
from slider import Slider
from heatmap import Heatmap

def main():

    # Video Capture
    cap = cv2.VideoCapture("video.mp4")
    width, height = int(cap.get(3)), int(cap.get(4))
    # Video Writer
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (width, height))
    # Create heatmap
    heatmap = Heatmap(width, height)
    # Create slider
    model = get_model("model.pickle")
    slider = Slider(model)
    # Read video frame
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Sliding window approach
            bounding_boxes = slider.get_bounding_boxes(frame)
            # Update heatmap
            heatmap.update(bounding_boxes)
            # Draw bounding boxes
            for x1, y1, x2, y2 in heatmap.get_bounding_boxes():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # Save output video
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release video capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

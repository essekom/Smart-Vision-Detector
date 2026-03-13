import cv2
from ultralytics import YOLO

class VisionDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the YOLOv8 model."""
        self.model = YOLO(model_path)

    def detect_realtime(self):
        """Run real-time detection via webcam."""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("Smart Vision - YOLOv8 Detection", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = VisionDetector()
    print("Starting Vision Detector... Press 'q' to quit.")
    detector.detect_realtime()

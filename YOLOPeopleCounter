import cv2
from ultralytics import YOLO
from djitellopy import Tello

# Load the YOLOv10 model
model = YOLO('yolov10s.pt')  # Ensure the correct path to the model file

# Initialize Tello drone
tello = Tello()

# Connect to Tello drone
tello.connect()
print(f"Connected to Tello drone. Battery Life Percentage: {tello.get_battery()}%")

# Start video stream
tello.streamon()

# Get the frame reader
frame_read = tello.get_frame_read()

# Get class names (assuming COCO dataset)
class_names = model.names

while True:
    # Get frame from Tello
    frame = frame_read.frame

    if frame is None:
        continue  # Skip iteration if no frame is received

    # Fix color issue by converting from RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(frame)

    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Initialize people count
    num_people = 0

    # Iterate over detections and process only 'person' class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls)
            if cls_id == 0:  # Class ID 0 corresponds to 'person'
                num_people += 1
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box on the frame
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),  # Green color for bounding box
                    2
                )
                # Put class name text above the bounding box
                cv2.putText(
                    annotated_frame,
                    class_names[cls_id],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

    # Add the count of people to the annotated frame
    cv2.putText(
        annotated_frame,
        f'People Count: {num_people}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),  # Green color text
        2
    )

    # Display the annotated frame
    cv2.imshow('YOLOv10 Object Detection', annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
tello.streamoff()
cv2.destroyAllWindows()

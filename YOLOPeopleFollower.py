import cv2
from ultralytics import YOLO
from djitellopy import Tello
import time

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

# Drone control parameters
forward_backward_velocity = 0
left_right_velocity = 0
up_down_velocity = 0
yaw_velocity = 0

# PID controllers for smooth movement
pid_yaw = [0.4, 0.0, 0.2]  # PID coefficients for yaw control
previous_error_yaw = 0

# Takeoff
tello.takeoff()
time.sleep(2)  # Give the drone time to stabilize

try:
    while True:
        # Get frame from Tello
        frame = frame_read.frame

        if frame is None:
            continue  # Skip iteration if no frame is received

        # Fix color issue by converting from RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Update frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Perform object detection
        results = model(frame)

        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()

        # Initialize variables
        num_people = 0
        nearest_person = None
        max_area = 0

        # Center of the frame
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2

        # Iterate over detections and process only 'person' class
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                if cls_id == 0:  # Class ID 0 corresponds to 'person'
                    num_people += 1
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Calculate area of the bounding box
                    area = (x2 - x1) * (y2 - y1)
                    # Update nearest person (largest bounding box area)
                    if area > max_area:
                        max_area = area
                        nearest_person = {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2,
                            'area': area
                        }
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

        # Control the drone to follow the nearest person
        if nearest_person is not None:
            # Calculate errors
            error_yaw = nearest_person['center_x'] - frame_center_x

            # PID control for yaw (left/right rotation)
            yaw_velocity = pid_yaw[0] * error_yaw + pid_yaw[1] * (error_yaw - previous_error_yaw)
            yaw_velocity = int(max(min(yaw_velocity, 100), -100))  # Constrain to [-100, 100]
            previous_error_yaw = error_yaw

            # Move the drone forward to approach the person
            forward_backward_velocity = 20  # Constant speed to move forward

            # Send velocities to the drone
            tello.send_rc_control(
                left_right_velocity=0,
                forward_backward_velocity=forward_backward_velocity,
                up_down_velocity=0,
                yaw_velocity=yaw_velocity
            )

            # Draw a circle at the center of the nearest person
            cv2.circle(
                annotated_frame,
                (nearest_person['center_x'], nearest_person['center_y']),
                5,
                (0, 0, 255),  # Red color
                -1
            )

        else:
            # If no person detected, hover in place
            tello.send_rc_control(0, 0, 0, 0)

        # Display the annotated frame
        cv2.imshow('YOLOv10 Object Detection', annotated_frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Land the drone
    tello.land()
    # Release resources
    tello.streamoff()
    cv2.destroyAllWindows()

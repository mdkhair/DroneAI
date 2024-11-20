import cv2
import mediapipe as mp
from djitellopy import Tello
import threading
import time

# Initialize the Tello drone
tello = Tello()
tello.connect()
print(f"Connected to Tello drone. Battery Life Percentage: {tello.get_battery()}%")

# Start the video stream
tello.streamon()

# Initialize MediaPipe Hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global variables for frame and control
frame = None
stop_event = threading.Event()

def video_capture():
    global frame
    while not stop_event.is_set():
        frame = tello.get_frame_read().frame
        time.sleep(0.03)  # Adjust sleep time as needed

def process_frame():
    global frame
    while not stop_event.is_set():
        if frame is not None:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and find hand landmarks
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Count fingers
                    landmarks = hand_landmarks.landmark
                    fingers_count = count_fingers(landmarks)

                    # Display the finger count on the frame
                    cv2.putText(frame, f'Fingers: {fingers_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Control the Tello based on the number of raised fingers
                    if fingers_count == 5:
                        tello.land()
                        stop_event.set()  # Stop threads after landing
                    elif fingers_count == 1:
                        # Calculate the x-coordinate error between the finger and the center of the frame
                        frame_height, frame_width, _ = frame.shape
                        finger_x = int(landmarks[8].x * frame_width)  # x-coordinate of the index fingertip
                        error_x = finger_x - (frame_width // 2)  # Error relative to the center of the frame

                        # Adjust yaw velocity based on the error
                        yaw_velocity = int(error_x * 0.1)  # Scale the error to get an appropriate yaw velocity
                        yaw_velocity = max(min(yaw_velocity, 100), -100)  # Constrain yaw velocity to [-100, 100]

                        # Move forward and adjust yaw to follow the finger
                        tello.send_rc_control(0, 20, 0, yaw_velocity)  # Move forward and adjust yaw
                    elif fingers_count == 2 or fingers_count == 0:
                        # Hover in place
                        tello.send_rc_control(0, 0, 0, 0)  # Stop all movement

            # Display the frame
            cv2.imshow("Finger Gesture Control", frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()

# Function to count raised fingers with improved thumb detection
def count_fingers(landmarks):
    fingers = []

    # Thumb: Check if it's raised by comparing with the palm landmark (landmark[2])
    if landmarks[4].x > landmarks[3].x and landmarks[4].x > landmarks[2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Four fingers
    for tip in [8, 12, 16, 20]:
        if landmarks[tip].y < landmarks[tip - 2].y:  # Finger tip is above its middle joint
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Takeoff initially
tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)  # Hover in place

# Start threads
capture_thread = threading.Thread(target=video_capture)
process_thread = threading.Thread(target=process_frame)

capture_thread.start()
process_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()

# Land the drone and release resources
tello.land()
tello.streamoff()
cv2.destroyAllWindows()

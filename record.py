import cv2
import mediapipe as mp
import numpy as np
import subprocess

# Initialize mediapipe face detection and face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Use the external webcam by specifying the camera index (usually 0)
cap = cv2.VideoCapture(0)

# Define a narrow color range for dark shades (black and dark gray)
lower_dark = np.array([0, 0, 0], dtype=np.uint8)
upper_dark = np.array([180, 255, 70], dtype=np.uint8)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables to store the previous and current status
previous_status = None
current_status = "No Headphones"  # Initialize as "No Headphones"

def send_notification(title, message):
    """Send a notification using osascript (AppleScript)."""
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and landmarks in the frame
    results = face_detection.process(rgb_frame)
    face_landmarks = face_mesh.process(rgb_frame).multi_face_landmarks

    if results.detections and face_landmarks:
        # Select the largest detected face
        largest_detection = max(results.detections, key=lambda detection: detection.location_data.relative_bounding_box.width * detection.location_data.relative_bounding_box.height)
        bboxC = largest_detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face landmarks
        landmarks = face_landmarks[0]
        right_ear_landmark = landmarks.landmark[234]  # Right ear landmark

        # Get coordinates for the right ear
        right_ear_x = int(right_ear_landmark.x * iw)
        right_ear_y = int(right_ear_landmark.y * ih)
        cv2.circle(frame, (right_ear_x, right_ear_y), 5, (0, 255, 0), -1)

        # Define a region around the right ear to check for headphones
        roi_start_x = max(0, right_ear_x - int(w * 0.25))
        roi_end_x = min(iw, right_ear_x + int(w * 0.25))
        roi_start_y = max(0, right_ear_y - int(h * 0.25))
        roi_end_y = min(ih, right_ear_y + int(h * 0.25))

        right_ear_region = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

        # Ensure the ROI is valid and not empty
        if right_ear_region.size == 0:
            current_status = "No Headphones"
        else:
            # Convert the region to HSV color space for color detection
            right_hsv = cv2.cvtColor(right_ear_region, cv2.COLOR_BGR2HSV)

            # Create mask for dark colors (black and dark gray)
            dark_mask = cv2.inRange(right_hsv, lower_dark, upper_dark)

            # Calculate the percentage of the area that is dark
            dark_percentage = np.sum(dark_mask) / dark_mask.size

            # Determine if headphones are detected based on the dark color percentage
            if dark_percentage > 0.2:  # Adjust threshold as needed
                current_status = "Headphones Detected"
            else:
                current_status = "No Headphones"

    else:
        current_status = "No Headphones"

    if previous_status != current_status:
        print("Status mudou para:", current_status)
        # Send notification
        send_notification("Headphone Status Changed", f"The status is now: {current_status}")

    previous_status = current_status

    # Display the current status on the frame
    if current_status == "Headphones Detected":
        cv2.putText(frame, 'Headphones Detected', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Headphones', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the resulting frame with headphone status
    cv2.imshow('Headphone Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

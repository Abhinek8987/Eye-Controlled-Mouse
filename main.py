import cv2
import mediapipe as mp
import pyautogui

# Initialize Camera
webcam = cv2.VideoCapture(0)

# Initialize FaceMesh
face_detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen resolution
screen_width, screen_height = pyautogui.size()

while True:
    # Read frame from webcam
    ret, frame = webcam.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    processed_frame = face_detector.process(rgb_frame)
    facial_landmarks = processed_frame.multi_face_landmarks

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    if facial_landmarks:
        # Extract landmarks
        keypoints = facial_landmarks[0].landmark

        # Iterate over specific eye landmarks for tracking
        for idx, point in enumerate(keypoints[474:478]):
            x_coord = int(point.x * frame_width)
            y_coord = int(point.y * frame_height)

            # Draw tracking points on the frame
            cv2.circle(frame, (x_coord, y_coord), 4, (0, 255, 0), -1)

            if idx == 1:
                cursor_x = screen_width * point.x
                cursor_y = screen_height * point.y
                pyautogui.moveTo(cursor_x, cursor_y)

        # Detect blink using landmarks
        left_eye_points = [keypoints[145], keypoints[159]]
        for eye_point in left_eye_points:
            eye_x = int(eye_point.x * frame_width)
            eye_y = int(eye_point.y * frame_height)
            cv2.circle(frame, (eye_x, eye_y), 4, (0, 255, 255), -1)

        # Blink detection logic
        if (left_eye_points[0].y - left_eye_points[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Display the output frame
    cv2.imshow('Eye Mouse Control', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
webcam.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np

# Emoji labels corresponding to detected facial expressions
EMOJI_LABELS = {
    "neutral": "NEUTRAL",
    "happy": "HAPPY",
    "sad": "SAD",
    "surprised": "SURPRISED",
    "angry": "ANGRY"
}

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_emoji = "neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw facial landmarks
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )

            landmarks = face_landmarks.landmark
            try:
                # @param chin, forehead: Used to normalize face proportions
                # Calculate normalized face height
                chin = np.array([landmarks[152].x, landmarks[152].y])
                forehead = np.array([landmarks[10].x, landmarks[10].y])
                face_height = np.linalg.norm(forehead - chin)

                # @param top_lip, bottom_lip: Used to detect mouth opening
                # Calculate mouth openness ratio
                top_lip = np.array([landmarks[13].x, landmarks[13].y])
                bottom_lip = np.array([landmarks[14].x, landmarks[14].y])
                mouth_ratio = np.linalg.norm(top_lip - bottom_lip) / face_height

                # @param upper_eyelid, lower_eyelid: Used to detect eye openness
                # Calculate eye openness ratio (left eye)
                upper_eyelid = np.array([landmarks[159].x, landmarks[159].y])
                lower_eyelid = np.array([landmarks[145].x, landmarks[145].y])
                eye_openness_ratio = np.linalg.norm(upper_eyelid - lower_eyelid) / face_height

                # @param left_brow, left_eye: Used to measure eyebrow raise
                # Calculate eyebrow to eye distance ratio
                left_brow = np.array([landmarks[55].x, landmarks[55].y])
                left_eye = np.array([landmarks[33].x, landmarks[33].y])
                brow_ratio = np.linalg.norm(left_brow - left_eye) / face_height

                # @param inner_left_brow, inner_right_brow: Used to detect frowning
                # Calculate distance between inner brows
                inner_left_brow = np.array([landmarks[70].x, landmarks[70].y])
                inner_right_brow = np.array([landmarks[300].x, landmarks[300].y])
                brow_center_distance = np.linalg.norm(inner_left_brow - inner_right_brow) / face_height

                # 🎯 Expression detection logic
                if mouth_ratio > 0.07 and eye_openness_ratio > 0.04:
                    current_emoji = "surprised"
                elif mouth_ratio > 0.035:
                    current_emoji = "happy"
                elif brow_center_distance < 0.04:
                    current_emoji = "angry"
                elif brow_ratio < 0.11:
                    current_emoji = "sad"
                else:
                    current_emoji = "neutral"

            except IndexError:
                # Skip frame if landmarks are not properly detected
                pass

    # Display the current detected emotion label on the frame
    label = EMOJI_LABELS[current_emoji]
    cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow("Smart Emoji Face", frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

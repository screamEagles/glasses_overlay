import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import os


# Load glasses with transparency
list_img = os.listdir("images")

img_list = []
for img_path in list_img:
    glasses_img = cv2.imread(f"./images/{img_path}", cv2.IMREAD_UNCHANGED)
    img_list.append(glasses_img)

index_img = 0


cap = cv2.VideoCapture("./video.mp4") # Replace the video with 0 to open the webcam

detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()

    # If video ends, reset to start
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to frame 0
        success, img = cap.read()  # Read again
        if not success:
            break  # Exit if still no frame

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        for face in faces:
            # Get left and right eye landmarks (points 33 and 263)
            left_eye = face[33]
            right_eye = face[263]

            # Calculate eye width and center x y positions
            eye_width = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 1.5)
            center_x = int((left_eye[0] + right_eye[0]) / 2)
            center_y = int((left_eye[1] + right_eye[1]) / 2)

            # Calculate angle between eyes (for rotation)
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Resize glasses while keeping aspect ratio
            aspect_ratio = img_list[index_img].shape[1] / img_list[index_img].shape[0]
            new_height = int(eye_width / aspect_ratio)
            glasses_resized = cv2.resize(img_list[index_img], (eye_width, new_height))

            # Rotate the glasses
            M = cv2.getRotationMatrix2D((eye_width // 2, new_height // 2), -angle, 1)
            glasses_rotated = cv2.warpAffine(
                glasses_resized, M, (eye_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )

            # Calculate position to place glasses
            x_offset = center_x - eye_width // 2
            y_offset = center_y - new_height // 2

            # Overlay glasses with transparency
            if x_offset >= 0 and y_offset >= 0 and x_offset + eye_width <= img.shape[1] and y_offset + new_height <= img.shape[0]:
                # Extract alpha channel and apply it
                alpha_mask = glasses_rotated[:, :, 3] / 255.0
                inverse_alpha = 1.0 - alpha_mask

                for c in range(0, 3):
                    img[y_offset:y_offset + new_height, x_offset:x_offset + eye_width, c] = (
                        inverse_alpha * img[y_offset:y_offset + new_height, x_offset:x_offset + eye_width, c] +
                        alpha_mask * glasses_rotated[:, :, c]
                    )
    
    cv2.imshow("Glasses Overlay", img)
    key = cv2.waitKey(1)
    if key == ord("a"):
        if index_img > 0:
            index_img -= 1
    if key == ord("d"):
        if index_img < len(img_list) - 1:
            index_img += 1
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

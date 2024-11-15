import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)

# Folder path with images
input_folder = '/users/oishideb/oishideb/all_mvdream_target_img_folders/Homer_hands_down_target_img/resized_images/'

# Folder to save annotated images
output_folder = '/users/oishideb/oishideb/all_mvdream_target_img_folders/mediapipe_results' 
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported image formats
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image from BGR to RGB as MediaPipe requires RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect keypoints
        results = pose.process(image_rgb)

        # Draw landmarks if keypoints are detected
        if results.pose_landmarks:
            # Draw the landmarks on the original image
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )
            
            # Save the annotated image
            annotated_image_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(annotated_image_path, annotated_image)
            print(f"Processed and saved: {annotated_image_path}")

            # Print landmark points
            print(f"Keypoints for {filename}:")
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                print(f"Landmark {i}: (x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility})")
        else:
            print(f"No keypoints detected in {filename}")

# Release resources
pose.close()
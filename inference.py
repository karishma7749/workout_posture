import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import os
import tensorflow as tf
import logging
import time
from ui_utils import (
	create_panel_background,
	add_modern_title,
	add_modern_info_box,
	add_footer,
	add_panel_borders,
	add_panel_separator,
	add_confidence_display
)

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)
logger = logging.getLogger('ExerciseAI')

# Suppress other frameworks' warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

def inFrame(lst):
	"""Check if enough landmarks are visible with very lenient threshold."""
	# Key landmarks for basic pose detection
	key_landmarks = [
		11, 12,  # shoulders
		13, 14,  # elbows
		23, 24   # hips
	]
	
	# Very low visibility threshold (30%)
	VISIBILITY_THRESHOLD = 0.3
	
	# Calculate average visibility of key landmarks
	visibility_scores = [lst[i].visibility for i in key_landmarks]
	avg_visibility = sum(visibility_scores) / len(visibility_scores)
	
	# Accept if average visibility is above threshold
	return avg_visibility > VISIBILITY_THRESHOLD

def calculate_form_quality(landmarks, exercise_name):
	"""Calculate form quality based on pose landmarks and exercise type"""
	quality_score = 0
	total_checks = 0
	
	# Get relevant joint angles
	def calculate_angle(p1, p2, p3):
		"""Calculate angle between three points"""
		v1 = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
		v2 = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
		cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
		angle = np.arccos(np.clip(cosine, -1.0, 1.0))
		return np.degrees(angle)

	# Common checks for symmetry
	left_shoulder = landmarks[11]
	right_shoulder = landmarks[12]
	left_hip = landmarks[23]
	right_hip = landmarks[24]
	
	# Check shoulder alignment
	shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
	if shoulder_diff < 0.1:  # Shoulders are level
		quality_score += 1
	total_checks += 1
	
	# Check hip alignment
	hip_diff = abs(left_hip.y - right_hip.y)
	if hip_diff < 0.1:  # Hips are level
		quality_score += 1
	total_checks += 1
	
	# Exercise-specific checks
	if "push" in exercise_name.lower():
		# Check elbow alignment for push-ups
		left_elbow = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
		right_elbow = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
		if abs(left_elbow - right_elbow) < 15:  # Elbows bend similarly
			quality_score += 1
		total_checks += 1
	
	elif "biceps" in exercise_name.lower():
		# Check arm positioning for bicep curls
		left_arm = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
		right_arm = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
		if abs(left_arm - right_arm) < 15:  # Arms move symmetrically
			quality_score += 1
		total_checks += 1
	
	# Calculate final quality score
	final_score = (quality_score / total_checks) if total_checks > 0 else 0
	
	# Convert score to qualitative assessment
	if final_score >= 0.9:
		return "Excellent"
	elif final_score >= 0.7:
		return "Good"
	elif final_score >= 0.5:
		return "Fair"
	else:
		return "Needs Improvement"

# Initialize display window
logger.info("Initializing ExerciseAI Professional...")
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
PANEL_WIDTH = int(WINDOW_WIDTH / 3)
main_window = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype="uint8")

# Load model and labels
logger.info("Loading AI model and labels...")
data_dir = "data/model"
model = load_model(os.path.join(data_dir, "model.h5"))
label = np.load(os.path.join(data_dir, "labels.npy"))
logger.info(f"Model loaded successfully. Available exercises: {len(label)}")
logger.info(f"Available exercise classes: {', '.join(label)}")  # Print all available classes

# Initialize MediaPipe
logger.info("Initializing pose detection pipeline...")
holistic = mp.solutions.pose
holis = holistic.Pose(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5,
	model_complexity=2
)
drawing = mp.solutions.drawing_utils

# Video setup
video_path = os.path.join(data_dir, "sample_video", "exercise.mp4")
logger.info(f"Opening video file: {video_path}")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
logger.info(f"Video properties - FPS: {fps}, Total Frames: {total_frames}")

# Output video setup
output_path = os.path.join(data_dir, "sample_video", "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (WINDOW_WIDTH, WINDOW_HEIGHT))
logger.info("Video writer initialized")

frame_count = 0
start_time = time.time()

while cap.isOpened():
	ret, frm = cap.read()
	if not ret:
		break

	frame_count += 1
	if frame_count % 30 == 0:  # Log every 30 frames
		elapsed_time = time.time() - start_time
		logger.info(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%) - {frame_count/elapsed_time:.1f} FPS")

	# Create modern panels
	left_panel = create_panel_background(PANEL_WIDTH, WINDOW_HEIGHT)
	middle_panel = create_panel_background(PANEL_WIDTH, WINDOW_HEIGHT)
	right_panel = create_panel_background(PANEL_WIDTH, WINDOW_HEIGHT)

	# Add modern titles
	add_modern_title(left_panel, "Live Video Feed")
	add_modern_title(middle_panel, "Pose Analysis")
	add_modern_title(right_panel, "Combined View")

	# Calculate display dimensions with margins
	display_height = int(WINDOW_HEIGHT * 0.65)
	display_width = int(PANEL_WIDTH * 0.9)
	margin_x = int(PANEL_WIDTH * 0.05)
	margin_y = 100  # Below title bar

	# Add panel borders
	for panel in [left_panel, middle_panel, right_panel]:
		add_panel_borders(panel, margin_x, margin_y, display_width, display_height)

	# Process and display frames
	input_frame = cv2.resize(frm, (display_width, display_height))
	left_panel[margin_y:margin_y+display_height, margin_x:margin_x+display_width] = input_frame

	# Process frame with MediaPipe
	rgb_frame = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
	res = holis.process(rgb_frame)

	if res.pose_landmarks:
		# Landmark visualization with modern styling
		landmark_frame = np.zeros((display_height, display_width, 3), dtype="uint8")
		drawing.draw_landmarks(
			landmark_frame, 
			res.pose_landmarks, 
			holistic.POSE_CONNECTIONS,
			drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
			drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)
		)
		middle_panel[margin_y:margin_y+display_height, margin_x:margin_x+display_width] = landmark_frame

		# Combined view with modern overlay
		overlay_frame = input_frame.copy()
		drawing.draw_landmarks(
			overlay_frame,
			res.pose_landmarks,
			holistic.POSE_CONNECTIONS,
			drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
			drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)
		)
		right_panel[margin_y:margin_y+display_height, margin_x:margin_x+display_width] = overlay_frame

		# Modern exercise analysis display
		if inFrame(res.pose_landmarks.landmark):
			lst = []
			for i in res.pose_landmarks.landmark:
				lst.append(i.x - res.pose_landmarks.landmark[0].x)
				lst.append(i.y - res.pose_landmarks.landmark[0].y)

			lst = np.array(lst).reshape(1,-1)
			p = model.predict(lst, verbose=0)
			
			# Debug: Print all predictions every 30 frames
			if frame_count % 30 == 0:
				logger.info("All predictions:")
				for idx, confidence in enumerate(p[0]):
					logger.info(f"  {label[idx]}: {confidence:.1%}")
			
			pred = label[np.argmax(p)]
			confidence = p[0][np.argmax(p)]

			# Lower confidence threshold for debugging
			if confidence > 0.5:
				if frame_count % 30 == 0:
					logger.info(f"Top detection: {pred} (Confidence: {confidence:.1%})")

				# Calculate form quality
				form_quality = calculate_form_quality(res.pose_landmarks.landmark, pred)

				info_dict = {
					"Exercise": pred.replace("_", " ").title(),
					"Confidence": f"{confidence:.1%}",
					"Form Quality": form_quality
				}

				info_box_start_y = WINDOW_HEIGHT - 450
				add_modern_info_box(right_panel, info_dict, info_box_start_y)
				
				# Display pose confidences in middle panel
				add_confidence_display(middle_panel, p[0], label, WINDOW_HEIGHT - 400)
			else:
				logger.info(f"Low confidence detection: {pred} ({confidence:.1%})")
				cv2.circle(middle_panel, (35, WINDOW_HEIGHT-50), 8, (0, 0, 255), -1)
				cv2.putText(middle_panel, "Status: Inactive", (55, WINDOW_HEIGHT-45), 
						  cv2.FONT_HERSHEY_DUPLEX, 0.7, (230, 230, 230), 1)
	else:
		# Handle case when no pose is detected
		cv2.putText(middle_panel, "No pose detected", (margin_x, margin_y + display_height + 30),
				   cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 50, 50), 1)

	# Combine panels with subtle separators
	main_window[:, 0:PANEL_WIDTH] = left_panel
	main_window[:, PANEL_WIDTH:PANEL_WIDTH*2] = middle_panel
	main_window[:, PANEL_WIDTH*2:] = right_panel
	
	# Add subtle panel separators
	add_panel_separator(main_window, PANEL_WIDTH, WINDOW_HEIGHT)
	add_panel_separator(main_window, PANEL_WIDTH*2, WINDOW_HEIGHT)

	# Modern footer
	add_footer(main_window, WINDOW_HEIGHT, WINDOW_WIDTH)

	cv2.imshow("ExerciseAI Professional", main_window)
	out.write(main_window)

	if cv2.waitKey(1) & 0xFF == 27:
		logger.info("User terminated processing")
		break

cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
logger.info(f"Analysis complete! Processed {frame_count} frames in {total_time:.1f} seconds")
logger.info(f"Average processing speed: {frame_count/total_time:.1f} FPS")
logger.info(f"Output saved to: {output_path}")


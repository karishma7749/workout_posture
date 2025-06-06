import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm  # For progress bars

# Define required landmarks by body areas
BODY_PARTS = {
	"CORE": {
		23: "LEFT_HIP",
		24: "RIGHT_HIP"
	},
	"UPPER_BODY": {
		11: "LEFT_SHOULDER",
		12: "RIGHT_SHOULDER",
		13: "LEFT_ELBOW",
		14: "RIGHT_ELBOW"
	},
	"HANDS": {
		15: "LEFT_WRIST",
		16: "RIGHT_WRIST"
	},
	"LEGS": {
		25: "LEFT_KNEE",
		26: "RIGHT_KNEE",
		27: "LEFT_ANKLE",
		28: "RIGHT_ANKLE"
	}
}

# Define required landmarks for each exercise
EXERCISE_REQUIREMENTS = {
	"barbell_biceps_curl": ["CORE", "UPPER_BODY", "HANDS"],
	"chest_fly_machine": ["CORE", "UPPER_BODY", "HANDS"],
	"deadlift": ["CORE", "UPPER_BODY", "HANDS", "LEGS"],
	"hammer_curl": ["UPPER_BODY", "HANDS"],
	"hip_thrust": ["CORE", "UPPER_BODY", "HANDS", "LEGS"],
	"lat_pulldown": ["CORE", "UPPER_BODY", "HANDS"],
	"leg_extension": ["CORE", "UPPER_BODY", "LEGS"],
	"push-up": ["CORE", "UPPER_BODY", "HANDS", "LEGS"],
	"plank": ["CORE", "UPPER_BODY", "HANDS", "LEGS"],
	"shoulder_press": ["CORE", "UPPER_BODY", "HANDS"],
	"russian_twist": ["CORE", "UPPER_BODY", "HANDS"],
	"bench_press": ["CORE", "UPPER_BODY", "HANDS"],
	"decline_bench_press": ["CORE", "UPPER_BODY", "HANDS"],
	"incline_bench_press": ["CORE", "UPPER_BODY", "HANDS"],
	"lateral_raise": ["CORE", "UPPER_BODY", "HANDS"],
	"leg_raises": ["CORE", "LEGS"],
	"pull_up": ["CORE", "UPPER_BODY", "HANDS"],
	"romanian_deadlift": ["CORE", "UPPER_BODY", "HANDS", "LEGS"],
	"squat": ["CORE", "UPPER_BODY", "LEGS"],
	"t_bar_row": ["CORE", "UPPER_BODY", "HANDS"],
	"tricep_pushdown": ["UPPER_BODY", "HANDS"],
	"tricep_dips": ["UPPER_BODY", "HANDS"]
}


SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mov')  # Added supported formats as constant

def get_required_landmarks(exercise_name):
	"""Get required landmarks for specific exercise."""
	required_parts = EXERCISE_REQUIREMENTS.get(exercise_name, ["CORE", "UPPER_BODY", "HANDS", "LEGS"])
	landmarks = {}
	for part in required_parts:
		landmarks.update(BODY_PARTS[part])
	return landmarks


VISIBILITY_THRESHOLD = 0.4

def inFrame(landmarks, required_landmarks):
	"""Check if required landmarks are visible above threshold."""
	return all(
		landmarks[idx].visibility > VISIBILITY_THRESHOLD 
		for idx in required_landmarks.keys()
	)

def process_video(video_path, holis, required_landmarks):
	"""Process a single video and extract pose data."""
	frame_data = []
	cap = cv2.VideoCapture(video_path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	try:
		for _ in tqdm(range(total_frames), desc="Processing frames"):
			ret, frm = cap.read()
			if not ret:
				break
				
			res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
			
			if res.pose_landmarks and inFrame(res.pose_landmarks.landmark, required_landmarks):
				# Extract normalized coordinates relative to nose position
				landmarks = res.pose_landmarks.landmark
				nose_pos = landmarks[0]
				pose_data = []
				for landmark in landmarks:
					pose_data.extend([
						landmark.x - nose_pos.x,
						landmark.y - nose_pos.y
					])
				frame_data.append(pose_data)
				
	except Exception as e:
		print(f"Error processing video {video_path}: {str(e)}")
	finally:
		cap.release()
		
	return frame_data

def main():
	# Initialize MediaPipe
	mp_pose = mp.solutions.pose
	holis = mp_pose.Pose(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)
	
	data_dir = "data"
	
	# Get list of exercise folders
	exercise_folders = [f for f in os.listdir(data_dir) 
					   if os.path.isdir(os.path.join(data_dir, f))]
	
	print(f"Found {len(exercise_folders)} exercise folders")
	
	for exercise_folder in tqdm(exercise_folders, desc="Processing exercises"):
		try:
			exercise_name = exercise_folder.replace(" ", "_").lower()
			required_landmarks = get_required_landmarks(exercise_name)
			print(f"\nProcessing {exercise_folder} using landmarks: {list(required_landmarks.values())}")
			
			# Normalize exercise name and create output path
			npy_path = os.path.join(data_dir, f"{exercise_name}.npy")
			
			# Skip if already processed
			if os.path.exists(npy_path):
				print(f"Skipping {exercise_folder} - NPY file already exists")
				continue
			
			video_folder = os.path.join(data_dir, exercise_folder)
			all_pose_data = []
			
			# Updated to include both .mp4 and .MOV files
			videos = [f for f in os.listdir(video_folder) 
					 if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)]
			print(f"\nProcessing {len(videos)} videos in {exercise_folder}...")
			
			for video_file in videos:
				video_path = os.path.join(video_folder, video_file)
				print(f"Processing {video_file}...")
				frame_data = process_video(video_path, holis, required_landmarks)
				all_pose_data.extend(frame_data)
				print(f"Extracted {len(frame_data)} frames from {video_file}")
			
			# Save collected data
			if all_pose_data:
				all_pose_data = np.array(all_pose_data)
				np.save(npy_path, all_pose_data)
				print(f"Saved {len(all_pose_data)} frames to {npy_path}")
				print(f"Data shape: {all_pose_data.shape}")
			else:
				print(f"No valid pose data collected for {exercise_folder}!")
				
		except Exception as e:
			print(f"Error processing {exercise_folder}: {str(e)}")
			
	holis.close()
	print("\nProcessing complete!")

if __name__ == "__main__":
	main()
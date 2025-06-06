import cv2
import os
from tqdm import tqdm
import numpy as np

# Add diagnostic information
print("OpenCV version:", cv2.getVersionString())
print("Available cv2 attributes:", dir(cv2))

SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mov')

def process_video(video_path, output_folder):
    """Process a single video and save frames as PNG."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
                
            frame_filename = f"frame_{frame_idx:04d}.png"
            frame_path = os.path.join(output_folder, frame_filename)
            
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"Failed to save frame: {frame_path}")
                
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()

def main():
    data_dir = "data"
    output_base_dir = "frame_data"  # Base directory for storing frames
    
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of exercise folders
    exercise_folders = [f for f in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, f))]
    
    print(f"Found {len(exercise_folders)} exercise folders")
    
    for exercise_folder in tqdm(exercise_folders, desc="Processing exercises"):
        try:
            exercise_name = exercise_folder.replace(" ", "_").lower()
            print(f"\nProcessing {exercise_folder}")
            
            # Create exercise-specific output directory
            exercise_output_dir = os.path.join(output_base_dir, exercise_name)
            os.makedirs(exercise_output_dir, exist_ok=True)
            
            video_folder = os.path.join(data_dir, exercise_folder)
            
            # Get all videos in the folder
            videos = [f for f in os.listdir(video_folder) 
                     if f.lower().endswith(SUPPORTED_VIDEO_FORMATS)]
            print(f"\nProcessing {len(videos)} videos in {exercise_folder}...")
            
            for video_file in videos:
                video_path = os.path.join(video_folder, video_file)
                
                # Create video-specific output directory
                video_name = os.path.splitext(video_file)[0]
                video_output_dir = os.path.join(exercise_output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                print(f"Processing {video_file}...")
                process_video(video_path, video_output_dir)
                print(f"Finished processing {video_file}")
                
        except Exception as e:
            print(f"Error processing {exercise_folder}: {str(e)}")
            
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

# Create a simple black image
img = np.zeros((100, 100, 3), dtype=np.uint8)

# If this runs without error, OpenCV is working
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
# Exercise Recognition Data Pipeline Documentation

## Overview

This pipeline processes exercise videos to create a standardized dataset for training a pose recognition model. The system is designed to be robust against variations in video quality, camera angles, and exercise performance styles.

## 1. Data Collection and Cleaning

### Video Input System

```python
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mov')
```

**Reasoning**:

- MP4 offers good compression while maintaining quality
- MOV format is common in iOS devices, ensuring broad device compatibility
- Both formats support high frame rates needed for exercise analysis

### Landmark System Architecture

```python
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
```

**Design Philosophy**:

1. Hierarchical Organization:

   - Grouped by body regions for logical access
   - Enables exercise-specific landmark selection
   - Facilitates future additions of new exercises

2. Landmark Selection Criteria:
   - Core landmarks chosen for stability measurement
   - Upper body points crucial for form analysis
   - Hand positions essential for grip tracking
   - Leg landmarks for lower body movement patterns

### Exercise-Specific Requirements

```python
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
    "russian_twist": ["CORE", "UPPER_BODY", "HANDS"]
}
```

**Reasoning for Requirements**:

1. Upper Body Exercises:

   - Barbell Biceps Curl: Core for posture, Upper Body and Hands for movement tracking
   - Chest Fly Machine: Core for stability, Upper Body and Hands for movement path
   - Hammer Curl: Focus on arm movement only
   - Shoulder Press: Core for stability, Upper Body and Hands for press movement

2. Full Body Exercises:

   - Deadlift: All landmarks for complete form analysis
   - Hip Thrust: Full body engagement monitoring
   - Push-up: Complete body alignment tracking
   - Plank: Full body position verification

3. Specialized Movements:

   - Lat Pulldown: Upper body focus with core stability
   - Leg Extension: Core stability with leg movement tracking
   - Russian Twist: Core movement with arm position tracking

## 2. Data Preprocessing Pipeline

### Quality Control System

```python
VISIBILITY_THRESHOLD = 0.4  # 40% confidence threshold

def inFrame(landmarks, required_landmarks):
    """Check if required landmarks are visible above threshold."""
    return all(
        landmarks[idx].visibility > VISIBILITY_THRESHOLD
        for idx in required_landmarks.keys()
    )
```

**Threshold Selection Logic**:

- 0.4 chosen as optimal balance between:
  - Data quality (higher = better quality)
  - Data quantity (lower = more frames)
  - Real-world conditions (lighting, occlusion)
  - Exercise variation accommodation

### Coordinate Normalization System

```python
def process_video(video_path, holis, required_landmarks):
    """Process a single video and extract pose data."""
    # Extract normalized coordinates relative to nose position
    landmarks = res.pose_landmarks.landmark
    nose_pos = landmarks[0]
    pose_data = []
    for landmark in landmarks:
        pose_data.extend([
            landmark.x - nose_pos.x,
            landmark.y - nose_pos.y
        ])
```

**Technical Reasoning**:

1. Nose Reference Point:

   - Stable central point
   - Usually visible in exercises
   - Natural center of coordinate system

2. Relative Positioning:
   - Makes data scale-invariant
   - Handles different video resolutions
   - Normalizes for subject height/position
   - Enables consistent model training

### Frame Processing Logic

```python
def process_video(video_path, holis, required_landmarks):
    """
    Multi-stage frame processing pipeline:
    1. Frame Extraction
    2. Quality Validation
    3. Data Normalization
    """
    frame_data = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frm = cap.read()
        if not ret:
            break

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark, required_landmarks):
            # Process frame data...
```

**Processing Stages**:

1. Frame Extraction:

   - Efficient video decoding
   - Memory-managed processing
   - Progress tracking with tqdm

2. Pose Detection:

   - MediaPipe integration
   - Real-time landmark detection
   - RGB color space conversion

3. Quality Validation:

   - Visibility threshold checking
   - Required landmark verification
   - Frame rejection if criteria not met

4. Data Normalization:
   - Coordinate system transformation
   - Scale normalization
   - Feature vector creation

## 3. Data Storage and Management

### NPY File Structure

```python
# Save collected data
if all_pose_data:
    all_pose_data = np.array(all_pose_data)
    np.save(npy_path, all_pose_data)
    print(f"Saved {len(all_pose_data)} frames to {npy_path}")
    print(f"Data shape: {all_pose_data.shape}")
```

**Design Decisions**:

1. File Format:

   - NPY chosen for:
     - Fast load times
     - Memory efficiency
     - NumPy compatibility
     - Preservation of array structure

2. Data Organization:
   - One file per exercise type
   - Consistent internal structure
   - Easy integration with training pipeline
   - Automatic skipping of already processed files

### Quality Assurance System

```python
def get_required_landmarks(exercise_name):
    """Get required landmarks for specific exercise."""
    required_parts = EXERCISE_REQUIREMENTS.get(exercise_name, ["CORE", "UPPER_BODY", "HANDS", "LEGS"])
    landmarks = {}
    for part in required_parts:
        landmarks.update(BODY_PARTS[part])
    return landmarks
```

**Validation Logic**:

1. Per-Frame Checks:

   - Individual landmark visibility
   - Required landmark presence
   - Threshold compliance
   - Exercise-specific requirements

2. Exercise-Specific Validation:
   - Dynamic landmark requirements
   - Body part grouping
   - Flexible fallback to full body tracking

## Output Specifications

### Data Structure

- Format: NumPy Arrays (.npy)
- Dimensions: [num_frames, num_features]
- Features: Normalized (x,y) coordinates relative to nose
- Organization: One file per exercise type
- Naming: Standardized exercise names (lowercase, underscores)

### Quality Metrics

1. Frame Level:

   - Landmark visibility scores
   - Required landmark validation
   - Position normalization

2. Exercise Level:
   - Frame count tracking
   - Data shape verification
   - Processing success logging

## Future Enhancements

1. Dynamic Threshold Adjustment:

   - Exercise-specific thresholds
   - Adaptive visibility requirements
   - Context-aware validation

2. Advanced Preprocessing:

   - Temporal smoothing
   - Noise reduction
   - Movement phase detection

3. Extended Validation:

   - Form correctness scoring
   - Rep counting
   - Exercise intensity estimation

4. Error Handling:
   - Improved error reporting
   - Recovery mechanisms
   - Data validation checks

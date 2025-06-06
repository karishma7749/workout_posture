import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set data directories
images_dir = "images"  # Updated directory name
data_dir = "data"
is_init = False
size = -1

def load_frames_from_exercise_dir(exercise_dir):
    """Load and preprocess frames from an exercise directory."""
    frames = []
    # Get all image files in the directory
    image_files = [f for f in os.listdir(exercise_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Use tqdm for progress tracking
    for frame_file in tqdm(image_files, desc=f"Loading {os.path.basename(exercise_dir)}"):
        frame_path = os.path.join(exercise_dir, frame_file)
        # Read image and resize to consistent dimensions
        frame = cv2.imread(frame_path)
        if frame is not None:
            frame = cv2.resize(frame, (224, 224))  # Standard size
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)
    
    return np.array(frames)

# Initialize lists for data and labels
X = []
y = []
label = []
dictionary = {}
c = 0

# Load frames from each exercise directory
print("Loading and preprocessing frames...")
for exercise_dir in sorted(os.listdir(images_dir)):
    exercise_path = os.path.join(images_dir, exercise_dir)
    if os.path.isdir(exercise_path):
        print(f"\nProcessing {exercise_dir}...")
        frames = load_frames_from_exercise_dir(exercise_path)
        
        if len(frames) > 0:
            if not is_init:
                is_init = True
                X = frames
                size = len(frames)
                y = np.array([exercise_dir] * size).reshape(-1, 1)
            else:
                X = np.concatenate((X, frames))
                y = np.concatenate((y, np.array([exercise_dir] * len(frames)).reshape(-1, 1)))

            label.append(exercise_dir)
            dictionary[exercise_dir] = c
            print(f"Loaded {len(frames)} frames from {exercise_dir}")
            c += 1

print(f"\nTotal exercises loaded: {len(label)}")
print("Exercises:", label)

# Convert labels to numeric indices
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert to one-hot encoding
y = to_categorical(y)

# Shuffle the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

print(f"\nTraining data shape: {X.shape}")
print(f"Number of classes: {y.shape[1]}")

# Build CNN model
input_shape = (224, 224, 3)  # Height, Width, Channels
ip = Input(shape=input_shape)

# CNN layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(ip)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Flatten and Dense layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
op = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=ip, outputs=op)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X, y,
    epochs=50,
    validation_split=0.2,
    batch_size=32
)

# Evaluate the model on the validation set
val_indices = int(X.shape[0] * 0.8)
val_loss, val_accuracy = model.evaluate(X[val_indices:], y[val_indices:])
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on the validation set
y_pred = model.predict(X[val_indices:])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y[val_indices:], axis=1)

# Compute and plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(label))
plt.xticks(tick_marks, label, rotation=45)
plt.yticks(tick_marks, label)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Save model and labels
model_dir = os.path.join(data_dir, "model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model_cnn2d.h5")
labels_path = os.path.join(model_dir, "labels_cnn2d.npy")

model.save(model_path)
np.save(labels_path, np.array(label))
print(f"\nModel saved to: {model_path}")
print(f"Labels saved to: {labels_path}") 
import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Set data directory
data_dir = "data"
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Load all .npy files from data directory
for i in os.listdir(data_dir):
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True 
			X = np.load(os.path.join(data_dir, i))
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			current_data = np.load(os.path.join(data_dir, i))
			X = np.concatenate((X, current_data))
			y = np.concatenate((y, np.array([i.split('.')[0]]*current_data.shape[0]).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		print(f"Loaded {i}: {current_data.shape[0] if 'current_data' in locals() else size} frames")
		c = c+1

print(f"\nTotal exercises loaded: {len(label)}")
print("Exercises:", label)

# Convert labels to numeric indices
for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert to one-hot encoding
y = to_categorical(y)

# Shuffle the data
X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1

print(f"\nTraining data shape: {X_new.shape}")
print(f"Number of classes: {y_new.shape[1]}")

# Build the model with correct input shape
ip = Input(shape=(X.shape[1],))
m = Dense(128, activation="tanh")(ip)
m = Dense(64, activation="tanh")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
print("\nTraining the model...")
history = model.fit(X_new, y_new, epochs=80, validation_split=0.2, batch_size=32)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_new[int(X_new.shape[0] * 0.8):], y_new[int(y_new.shape[0] * 0.8):])
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on the validation set
y_pred = model.predict(X_new[int(X_new.shape[0] * 0.8):])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_new[int(y_new.shape[0] * 0.8):], axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
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

# Print the classification report
target_names = label  # Use the exercise labels as target names
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

# Save model and labels in data/model directory
model_dir = os.path.join(data_dir, "model")
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

model_path = os.path.join(model_dir, "model.h5")
labels_path = os.path.join(model_dir, "labels.npy")

model.save(model_path)
np.save(labels_path, np.array(label))
print(f"\nModel saved to: {model_path}")
print(f"Labels saved to: {labels_path}")

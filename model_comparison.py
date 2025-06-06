import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def load_data(data_dir):
    """Load and prepare data from the data directory."""
    is_init = False
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
    
    return X, y, label, dictionary

def prepare_data(X, y, dictionary):
    """Prepare data for training."""
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
        
    return X_new, y_new

def create_dense_model(input_shape, num_classes):
    """Create the dense neural network model."""
    ip = Input(shape=(input_shape,))
    m = Dense(128, activation="tanh")(ip)
    m = Dense(64, activation="tanh")(m)
    op = Dense(num_classes, activation="softmax")(m)
    
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])
    return model

def create_cnn_model(input_shape, num_classes):
    """Create the CNN model."""
    ip = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation='relu')(ip)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    op = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=ip, outputs=op)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history1, history2, metric='loss'):
    """Plot training history comparison."""
    # Map common metric names
    metric_mapping = {
        'acc': 'accuracy',
        'accuracy': 'accuracy',
        'loss': 'loss'
    }
    
    metric_name = metric_mapping.get(metric, metric)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(history1.history[metric_name])
    plt.plot(history1.history[f'val_{metric_name}'])
    plt.plot(history2.history[metric_name])
    plt.plot(history2.history[f'val_{metric_name}'])
    plt.title(f'Model {metric_name}')
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend(['Dense Train', 'Dense Val', 'CNN Train', 'CNN Val'])
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(cm1, cm2, labels):
    """Plot confusion matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Dense model confusion matrix
    im1 = ax1.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("Dense Model Confusion Matrix")
    fig.colorbar(im1, ax=ax1)
    tick_marks = np.arange(len(labels))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Predicted Labels")
    ax1.set_ylabel("True Labels")
    
    # Plot CNN model confusion matrix
    im2 = ax2.imshow(cm2, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title("CNN Model Confusion Matrix")
    fig.colorbar(im2, ax=ax2)
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Predicted Labels")
    ax2.set_ylabel("True Labels")
    
    plt.tight_layout()
    plt.show()

def plot_model_metrics(dense_history, cnn_history):
    """Plot separate graphs for each model's metrics."""
    # Plot Dense model metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dense_history.history['loss'])
    plt.plot(dense_history.history['val_loss'])
    plt.title('Dense Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(dense_history.history['accuracy'])
    plt.plot(dense_history.history['val_accuracy'])
    plt.title('Dense Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.tight_layout()
    plt.show()
    
    # Plot CNN model metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['loss'])
    plt.plot(cnn_history.history['val_loss'])
    plt.title('CNN Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['accuracy'])
    plt.plot(cnn_history.history['val_accuracy'])
    plt.title('CNN Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.tight_layout()
    plt.show()

def main():
    # Set data directory
    data_dir = "data"
    
    # Load and prepare data
    X, y, label, dictionary = load_data(data_dir)
    X_new, y_new = prepare_data(X, y, dictionary)
    
    print(f"\nTotal exercises loaded: {len(label)}")
    print("Exercises:", label)
    print(f"\nTraining data shape: {X_new.shape}")
    print(f"Number of classes: {y_new.shape[1]}")
    
    # Prepare data for both models
    X_dense = X_new
    X_cnn = X_new.reshape(X_new.shape[0], -1, 2)
    
    # Create and train Dense model
    print("\nTraining Dense Model...")
    dense_model = create_dense_model(X_dense.shape[1], y_new.shape[1])
    dense_history = dense_model.fit(X_dense, y_new, epochs=80, validation_split=0.2, batch_size=32)
    
    # Create and train CNN model
    print("\nTraining CNN Model...")
    cnn_model = create_cnn_model((X_cnn.shape[1], 2), y_new.shape[1])
    cnn_history = cnn_model.fit(X_cnn, y_new, epochs=80, validation_split=0.2, batch_size=32)
    
    # Evaluate both models
    print("\nEvaluating Models...")
    dense_val_loss, dense_val_acc = dense_model.evaluate(
        X_dense[int(X_dense.shape[0] * 0.8):], 
        y_new[int(y_new.shape[0] * 0.8):]
    )
    cnn_val_loss, cnn_val_acc = cnn_model.evaluate(
        X_cnn[int(X_cnn.shape[0] * 0.8):], 
        y_new[int(y_new.shape[0] * 0.8):]
    )
    
    print("\nModel Comparison:")
    print(f"Dense Model - Validation Loss: {dense_val_loss:.4f}, Validation Accuracy: {dense_val_acc:.4f}")
    print(f"CNN Model - Validation Loss: {cnn_val_loss:.4f}, Validation Accuracy: {cnn_val_acc:.4f}")
    
    # Generate predictions and confusion matrices
    dense_pred = dense_model.predict(X_dense[int(X_dense.shape[0] * 0.8):])
    cnn_pred = cnn_model.predict(X_cnn[int(X_cnn.shape[0] * 0.8):])
    
    y_true = np.argmax(y_new[int(y_new.shape[0] * 0.8):], axis=1)
    dense_cm = confusion_matrix(y_true, np.argmax(dense_pred, axis=1))
    cnn_cm = confusion_matrix(y_true, np.argmax(cnn_pred, axis=1))
    
    # Plot model metrics
    print("\nPlotting model metrics...")
    plot_model_metrics(dense_history, cnn_history)
    
    # Plot confusion matrices
    plot_confusion_matrices(dense_cm, cnn_cm, label)
    
    # Print classification reports
    print("\nDense Model Classification Report:")
    print(classification_report(y_true, np.argmax(dense_pred, axis=1), target_names=label))
    
    print("\nCNN Model Classification Report:")
    print(classification_report(y_true, np.argmax(cnn_pred, axis=1), target_names=label))
    
    # Save models and labels
    model_dir = os.path.join(data_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    dense_model.save(os.path.join(model_dir, "model_dense.h5"))
    cnn_model.save(os.path.join(model_dir, "model_cnn.h5"))
    np.save(os.path.join(model_dir, "labels.npy"), np.array(label))
    
    print("\nModels and labels saved successfully!")

if __name__ == "__main__":
    main()
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def load_dataset(dataset_path):
    images = []
    labels = []
    
    for idx, blood_group in enumerate(BLOOD_GROUPS):
        blood_group_path = os.path.join(dataset_path, blood_group)
        if not os.path.exists(blood_group_path):
            print(f"Warning: {blood_group_path} does not exist")
            continue
            
        for image_name in os.listdir(blood_group_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(blood_group_path, image_name)
                try:
                    # Load and preprocess image
                    img = load_img(image_path, color_mode='grayscale', target_size=IMAGE_SIZE)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0  # Normalize
                    
                    images.append(img_array)
                    labels.append(idx)
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(BLOOD_GROUPS), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=BLOOD_GROUPS,
                yticklabels=BLOOD_GROUPS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/confusion_matrix.png')
    plt.close()

def main():
    # Create necessary directories
    os.makedirs('saved_model', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)
    
    # Load and preprocess dataset
    print("Loading dataset...")
    X, y = load_dataset('dataset')
    
    if len(X) == 0:
        print("No images found in the dataset!")
        return
    
    # Convert labels to categorical
    y_cat = to_categorical(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(X_test, y_test))
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=BLOOD_GROUPS))
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_test_classes, y_pred_classes)
    
    # Save model
    model.save('saved_model/fingerprint_model.h5')
    print("\nModel saved successfully!")

if __name__ == '__main__':
    main()
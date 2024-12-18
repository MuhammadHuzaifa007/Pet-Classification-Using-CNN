import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf
import streamlit as st

# Define helper functions
def read_images_and_labels_from_folder(folder_path, max_images=10000):
    """Read images and labels from the folder"""
    image_paths = []
    labels = []
    image_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')) and image_count < max_images:
                file_path = os.path.join(root, file)
                label = 0 if 'cat' in file.lower() else 1
                image_paths.append(file_path)
                labels.append(label)
                image_count += 1
            if image_count >= max_images:
                break
        if image_count >= max_images:
            break

    return image_paths, labels

def extract_features(image_paths):
    """Extract features from images"""
    features = []
    for path in tqdm(image_paths, desc="Processing Images"):
        try:
            image = Image.open(path)
            image = image.resize((128, 128), Image.LANCZOS)
            img_array = np.array(image)
            if img_array.shape == (128, 128, 3):  # Ensure consistent shape
                features.append(img_array)
        except Exception as e:
            st.error(f"Error processing {path}: {e}")

    features = np.array(features)
    features = features / 255.0  # Normalize
    return features

def extract_zip(zip_file_path, extract_to_path):
    """Extract zip file"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

# Streamlit UI
def main():
    st.title("Pet Classification Using CNN")

    # Upload zip file
    zip_file = st.file_uploader("Upload a dataset ZIP file:", type=['zip'])
    if zip_file:
        extract_to_path = 'dataset'
        os.makedirs(extract_to_path, exist_ok=True)

        # Extract dataset
        st.write("Extracting dataset...")
        extract_zip(zip_file, extract_to_path)

        # Read images and labels
        st.write("Reading images and labels...")
        image_paths, labels = read_images_and_labels_from_folder(extract_to_path, max_images=10000)

        # Extract features
        st.write("Extracting features from images...")
        X = extract_features(image_paths)
        y = np.array(labels)

        # Split dataset
        st.write("Splitting dataset...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Calculate class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        # Data Augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

        # Build Model
        st.write("Building and training the model...")
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        inputs = Input((128, 128, 3))
        x = base_model(inputs, training=True)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        # Callbacks
        checkpoint_path = 'best_model.keras'
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )

        # Train the model
        history = model.fit(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )

        # Plot training results
        st.write("Training complete! Displaying results...")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(acc, label='Training Accuracy')
        ax[0].plot(val_acc, label='Validation Accuracy')
        ax[0].set_title('Accuracy')
        ax[0].legend()

        ax[1].plot(loss, label='Training Loss')
        ax[1].plot(val_loss, label='Validation Loss')
        ax[1].set_title('Loss')
        ax[1].legend()

        st.pyplot(fig)

        # Predict on a single image
        image_index = st.slider("Select an image index for prediction:", 0, len(X)-1, 0)
        label_dict = {0: 'Cat', 1: 'Dog'}

        st.write("Original Label:", label_dict[y[image_index]])

        pred = model.predict(X[image_index].reshape(1, 128, 128, 3))
        pred_label = label_dict[round(pred[0][0])]
        st.write("Predicted Label:", pred_label)

        st.image(X[image_index], caption=f"Predicted: {pred_label}", use_column_width=True)

if __name__ == "__main__":
    main()

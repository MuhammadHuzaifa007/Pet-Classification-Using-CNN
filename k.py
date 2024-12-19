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

def read_images_and_labels_from_folder(folder_path, max_images=10000):
    """Read first and last N images and their labels from the folder."""
    image_paths = []
    labels = []

    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                all_files.append(os.path.join(root, file))

    all_files.sort()

    selected_files = all_files[:max_images] + all_files[-max_images:]

    for file_path in selected_files:
        label = 0 if 'cat' in file_path.lower() else 1
        image_paths.append(file_path)
        labels.append(label)

    return image_paths, labels

def extract_features(image_paths):
    """Extract features from images"""
    features = []
    for path in tqdm(image_paths, desc="Processing Images"):
        try:
            image = Image.open(path).convert('RGB')
            image = image.resize((128, 128), Image.LANCZOS)
            img_array = np.array(image)
            if img_array.shape == (128, 128, 3):
                features.append(img_array)
        except Exception as e:
            st.write(f"Error processing {path}: {e}")

    features = np.array(features)
    features = features / 255.0
    return features

def extract_zip(zip_file_path, extract_to_path):
    """Extract zip file"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

def build_model():
    """Build the CNN model"""
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    st.title("Pet Classification: Cat vs Dog")

    zip_file = st.file_uploader("Upload Dataset ZIP File", type=['zip'])
    if zip_file is not None:
        extract_to_path = "dataset"
        os.makedirs(extract_to_path, exist_ok=True)

        with st.spinner("Extracting ZIP file..."):
            extract_zip(zip_file, extract_to_path)

        st.success("Dataset extracted successfully!")

        st.write("Reading images and labels...")
        image_paths, labels = read_images_and_labels_from_folder(extract_to_path, max_images=10000)

        st.write("Extracting features...")
        X = extract_features(image_paths)
        y = np.array(labels)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        train_datagen = ImageDataGenerator(
            rotation_range=30,
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

        st.write("Building model...")
        model = build_model()

        checkpoint_path = 'best_model.keras'
        checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)

        with st.spinner("Training model..."):
            history = model.fit(train_generator, epochs=30, validation_data=val_generator, class_weight=class_weights, callbacks=[checkpoint, early_stopping, reduce_lr], verbose=1)

        st.success("Model trained successfully!")

        st.write("Plotting training results...")

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(history.history['accuracy'], label='Training Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].legend()
        ax[0].set_title("Accuracy")

        ax[1].plot(history.history['loss'], label='Training Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].legend()
        ax[1].set_title("Loss")

        st.pyplot(fig)

        model = load_model(checkpoint_path)

        st.write("Prediction on new image")
        test_image = st.file_uploader("Upload a test image", type=['jpg', 'jpeg', 'png'])

        if test_image is not None:
            try:
                image = Image.open(test_image).convert('RGB')
                image = image.resize((128, 128), Image.LANCZOS)
                img_array = np.array(image) / 255.0

                if img_array.shape == (128, 128, 3):
                    pred = model.predict(img_array.reshape(1, 128, 128, 3))
                    label_dict = {0: 'Cat', 1: 'Dog'}
                    pred_label = label_dict[round(pred[0][0])]

                    st.write(f"Predicted Label: {pred_label}")

                    fig, ax = plt.subplots()
                    ax.imshow(img_array)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.error("Invalid image shape!")
            except Exception as e:
                st.error(f"Error predicting image: {e}")

if __name__ == "__main__":
    main()

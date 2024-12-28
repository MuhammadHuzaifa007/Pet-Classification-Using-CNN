import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Function to read and preprocess images
def read_images_and_labels_from_folder(folder_path, img_size=(128, 128)):
    features = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                try:
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path).resize(img_size, Image.LANCZOS)
                    img_array = np.array(image)
                    if img_array.shape == (img_size[0], img_size[1], 3):
                        features.append(img_array / 255.0)  # Normalize
                        label = 0 if 'cat' in file.lower() else 1
                        labels.append(label)
                except Exception as e:
                    st.error(f"Error reading {file}: {e}")
    return np.array(features), np.array(labels)

# Function to build and compile the model
def build_model(input_shape=(128, 128, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(input_shape)
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess uploaded image
def preprocess_image(image, img_size=(128, 128)):
    try:
        image = image.resize(img_size, Image.LANCZOS)
        img_array = np.array(image)
        if img_array.shape == (img_size[0], img_size[1], 3):
            img_array = img_array / 255.0  # Normalize
            return np.expand_dims(img_array, axis=0)
        else:
            st.error("Invalid image shape! Please upload a 3-channel (RGB) image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
    return None

# Streamlit app
def main():
    st.title("Pet Classification")
    st.sidebar.title("Options")

    option = st.sidebar.selectbox("Choose an option", ["Train Model", "Predict Image"])
    
    if option == "Train Model":
        st.header("Train the Model")
        dataset_path = st.text_input("Enter the dataset folder path:")
        if dataset_path and st.button("Start Training"):
            X, y = read_images_and_labels_from_folder(dataset_path)
            if len(X) == 0:
                st.error("No valid images found in the dataset path.")
                return
            
            # Split dataset
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Calculate class weights
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(class_weights))
            
            # Data Augmentation
            train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                               shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
            val_datagen = ImageDataGenerator()
            train_generator = train_datagen.flow(X_train, y_train, batch_size=16)
            val_generator = val_datagen.flow(X_val, y_val, batch_size=16)
            
            # Build and train model
            model = build_model(input_shape=(128, 128, 3))
            checkpoint_path = 'cat_dog_classifier.keras'
            callbacks = [
                ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
            ]
            
            st.write("Training the model. Please wait...")
            history = model.fit(train_generator, validation_data=val_generator, epochs=20,
                                class_weight=class_weights, callbacks=callbacks)
            
            st.success("Training complete! Model saved as 'cat_dog_classifier.keras'.")
    
    elif option == "Predict Image":
        st.header("Predict an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            preprocessed_img = preprocess_image(image, img_size=(128, 128))
            if preprocessed_img is not None:
                try:
                    model = load_model('cat_dog_classifier.keras')
                    prediction = model.predict(preprocessed_img)
                    label = "Dog" if prediction[0][0] > 0.5 else "Cat"
                    st.success(f"The model predicts: {label}")
                    st.write(f"Confidence: {prediction[0][0]:.2f}")
                except Exception as e:
                    st.error(f"Error loading model: {e}")

if __name__ == "__main__":
    main()

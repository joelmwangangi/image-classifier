import streamlit as st
import os
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------------------
# 1. Drink image URLs & folder
# -------------------------------
image_urls = {
    "Coca-Cola": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Coca-Cola_logo.png",
    "Pepsi": "https://th.bing.com/th/id/OIP.73fMAA7nhyd4hHPo3eizYwHaJQ?w=149&h=187&c=7&r=0&o=7&pid=1.7&rm=3",
    "Red Bull": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Red_Bull.svg.png",
    "Sprite": "https://upload.wikimedia.org/wikipedia/commons/1/19/Sprite_logo_2019.svg.png",
    "Fanta": "https://upload.wikimedia.org/wikipedia/commons/1/11/Fanta_logo.svg.png",
    "Monster": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Monster_Energy_logo.svg.png",
    "Minute Maid": "https://upload.wikimedia.org/wikipedia/commons/0/05/Minute_Maid_logo.svg.png",
    "Dasani": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Dasani_logo.svg.png",
    "Lipton": "https://upload.wikimedia.org/wikipedia/commons/0/02/Lipton_Logo_2018.png",
    "Milo": "https://upload.wikimedia.org/wikipedia/commons/7/77/Nestle_Milo_logo.png"
}

image_folder = "./images"
os.makedirs(image_folder, exist_ok=True)

# -------------------------------
# 2. Download images robustly
# -------------------------------
st.write("🔄 Checking and downloading images...")
for drink, url in image_urls.items():
    img_path = os.path.join(image_folder, f"{drink}.jpg")
    if not os.path.exists(img_path):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, format="JPEG")
            st.write(f"✅ Downloaded {drink}")
        except Exception as e:
            st.write(f"❌ Failed to download {drink}: {e}")

# -------------------------------
# 3. Dataset folders
# -------------------------------
base_dir = "./dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

for dir_path in [train_dir, val_dir]:
    os.makedirs(dir_path, exist_ok=True)
    for drink in image_urls.keys():
        drink_train_dir = os.path.join(train_dir, drink)
        drink_val_dir = os.path.join(val_dir, drink)
        os.makedirs(drink_train_dir, exist_ok=True)
        os.makedirs(drink_val_dir, exist_ok=True)

        src = os.path.join(image_folder, f"{drink}.jpg")
        if os.path.exists(src):
            dest_train = os.path.join(drink_train_dir, f"{drink}_1.jpg")
            dest_val = os.path.join(drink_val_dir, f"{drink}_1.jpg")
            if not os.path.exists(dest_train):
                Image.open(src).save(dest_train)
            if not os.path.exists(dest_val):
                Image.open(src).save(dest_val)
        else:
            st.write(f"⚠️ Skipping {drink} — source image not found.")

# -------------------------------
# 4. Image Data Generators
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128,128), batch_size=2, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128,128), batch_size=2, class_mode='categorical'
)

# -------------------------------
# 5. Build CNN
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

st.title("🍹 Drink Classifier & Search App")

# -------------------------------
# 6. Train model button
# -------------------------------
if st.button("Train Model"):
    if train_generator.samples == 0:
        st.error("No training images available. Please check downloads.")
    else:
        st.write("Training model...")
        history = model.fit(train_generator, epochs=5, validation_data=val_generator)
        st.success("✅ Model Trained!")

        # Accuracy/Loss
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].plot(history.history['accuracy'], label='train_acc')
        ax[0].plot(history.history['val_accuracy'], label='val_acc')
        ax[0].set_title("Accuracy")
        ax[0].legend()
        ax[1].plot(history.history['loss'], label='train_loss')
        ax[1].plot(history.history['val_loss'], label='val_loss')
        ax[1].set_title("Loss")
        ax[1].legend()
        st.pyplot(fig)

        # Confusion matrix
        if val_generator.samples > 0 and len(train_generator.class_indices) > 0:
            val_generator.reset()
            y_pred = model.predict(val_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = val_generator.classes
            class_labels = list(train_generator.class_indices.keys())
            if len(class_labels) == len(np.unique(y_true)):
                cm = confusion_matrix(y_true, y_pred_classes)
                disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
                fig_cm, ax_cm = plt.subplots(figsize=(6,6))
                disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
                st.pyplot(fig_cm)
            else:
                st.write("⚠️ Skipping confusion matrix: class labels mismatch")

        model.save("drink_classifier.h5")
        st.success("Model saved as drink_classifier.h5")

# -------------------------------
# 7. Search & Predict
# -------------------------------
st.subheader("🔎 Search Drink & Predict")
drink_name = st.text_input("Enter drink name:")

if drink_name:
    img_path = os.path.join(image_folder, f"{drink_name}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=drink_name, use_column_width=True)

       

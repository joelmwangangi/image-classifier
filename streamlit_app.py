# streamlit_app.py
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from io import BytesIO
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -------------------------------
# 1. Setup Dataset and Image URLs
# -------------------------------
image_urls = {
    "Coca-Cola": "https://i5.walmartimages.com/asr/e3e510eb-3379-4ce5-a8e2-31f45ed5a47e.c99ca9cb61a8a839c23892605149d63b.jpeg",
    "Pepsi": "https://upload.wikimedia.org/wikipedia/commons/0/09/Pepsi_logo_2014.png",
    "Red Bull": "https://tse2.mm.bing.net/th/id/OIP.Kjony-rkRzNezIKBM3MSVwHaHa?rs=1&pid=ImgDetMain&o=7&rm=3",
    "Sprite": "https://www.coca-cola.com/content/dam/onexp/mv/en/brands/sprite/images/Sprite-desktop.png",
    "Fanta": "https://www.coca-cola.com/content/dam/onexp/mv/home-images/fanta/Fanta-desktop.png",
    "Monster": "https://www.instacart.com/image-server/1398x1398/www.instacart.com/assets/domains/product-image/file/large_fe53f5ed-45d0-4c95-80a3-c08387ef11c3.png",
    "Minute Maid": "https://th.bing.com/th/id/OIP.dYD6ZHQDJtZBAnNhX_GXVwHaHa?w=215&h=215&c=7&r=0&o=7&pid=1.7&rm=3",
    "Dasani": "https://th.bing.com/th/id/OIP.kY0UCtFI8O96ZgOd7YVZEwHaHa?w=202&h=202&c=7&r=0&o=7&pid=1.7&rm=3",
    "Lipton": "https://th.bing.com/th/id/OIP.llaNAJhqtzl76J0-WCKzWQHaHa?w=207&h=207&c=7&r=0&o=7&pid=1.7&rm=3",
    "Milo": "https://th.bing.com/th/id/OIP.u7w2lXk5w_rdWlul0Po1vAHaHa?w=215&h=215&c=7&r=0&o=7&pid=1.7&rm=3"
}

# Pre-downloaded images folder
image_folder = "./images"
os.makedirs(image_folder, exist_ok=True)

# Download images if not already present
for drink, url in image_urls.items():
    img_path = os.path.join(image_folder, f"{drink}.jpg")
    if not os.path.exists(img_path):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content))
            img.save(img_path)
        except:
            print(f"Failed to download {drink}")

# -------------------------------
# 2. Dataset Setup (train/val)
# -------------------------------
base_dir = "./dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

for dir_path in [train_dir, val_dir]:
    os.makedirs(dir_path, exist_ok=True)
    for drink in image_urls.keys():
        os.makedirs(os.path.join(dir_path, drink), exist_ok=True)
        # Copy the same image to train/val for demo
        src = os.path.join(image_folder, f"{drink}.jpg")
        if os.path.exists(src):
            dest_train = os.path.join(train_dir, drink, f"{drink}_1.jpg")
            dest_val = os.path.join(val_dir, drink, f"{drink}_1.jpg")
            if not os.path.exists(dest_train):
                Image.open(src).save(dest_train)
            if not os.path.exists(dest_val):
                Image.open(src).save(dest_val)

# -------------------------------
# 3. Image Data Generators
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
# 4. Build CNN model
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

st.title("Drink Classifier & Search App")

# -------------------------------
# 5. Train model (Optional)
# -------------------------------
if st.button("Train Model"):
    history = model.fit(train_generator, epochs=5, validation_data=val_generator)
    st.success("Model Trained!")

    # Accuracy/Loss plots
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
    val_generator.reset()
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(cm, display_labels=val_generator.class_indices.keys())
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot(plt.gcf())

    model.save("drink_classifier.h5")
    st.success("Model saved as drink_classifier.h5")

# -------------------------------
# 6. Drink Search
# -------------------------------
st.subheader("Search Drink Image")
drink_name = st.text_input("Enter drink name:")

if drink_name:
    img_path = os.path.join(image_folder, f"{drink_name}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, caption=drink_name, use_column_width=True)
    else:
        st.error(f"No image found for '{drink_name}'")

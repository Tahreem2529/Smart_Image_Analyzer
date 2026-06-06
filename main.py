import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Data
train_data = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2
)
test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    'dataset/train', target_size=(128,128),
    batch_size=16, class_mode='categorical'
)
test_generator = test_data.flow_from_directory(
    'dataset/test', target_size=(128,128),
    batch_size=16, class_mode='categorical'
)

# Transfer Learning Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# Accuracy Graph
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Loss Graph
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Dominant Colors Functions
def get_dominant_colors(image_path, n_colors=5):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = counts / counts.sum() * 100
    return colors, percentages

def show_dominant_colors(image_path):
    colors, percentages = get_dominant_colors(image_path)
    sorted_idx = np.argsort(percentages)[::-1]
    colors = colors[sorted_idx]
    percentages = percentages[sorted_idx]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    for i, (color, pct) in enumerate(zip(colors, percentages)):
        ax2.bar(i, pct, color=np.array(color)/255,
                edgecolor='black', width=0.6)
        ax2.text(i, pct + 0.5, f'{pct:.1f}%',
                ha='center', fontsize=10)
        ax2.text(i, -3, f'RGB\n{color[0]},{color[1]},{color[2]}',
                ha='center', fontsize=8)
    ax2.set_title('Dominant Colors')
    ax2.set_ylabel('Percentage %')
    ax2.set_xticks(range(len(colors)))
    ax2.set_xticklabels([f'Color {i+1}' for i in range(len(colors))])
    plt.tight_layout()
    plt.show()

# Analyze Function (dataset images ke liye)
def analyze_image(image_path):
    print("=" * 40)
    print("   SMART IMAGE ANALYZER")
    print("=" * 40)
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(128,128)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    classes = ['Cats', 'Dogs', 'Horses']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"\n Animal: {predicted_class}")
    print(f" Confidence: {confidence:.2f}%")
    print("\n Dominant Colors:")
    show_dominant_colors(image_path)

# GUI
def upload_and_analyze():
    file_path = filedialog.askopenfilename(
        title="Image Select Karo",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    img = tf.keras.preprocessing.image.load_img(
        file_path, target_size=(128,128)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    classes = ['Cats', 'Dogs', 'Horses']
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    result_label.config(
        text=f"Animal: {predicted_class}\nConfidence: {confidence:.2f}%"
    )
    display_img = Image.open(file_path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(display_img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    colors, percentages = get_dominant_colors(file_path)
    sorted_idx = np.argsort(percentages)[::-1]
    colors = colors[sorted_idx]
    percentages = percentages[sorted_idx]
    for i, (color, pct) in enumerate(zip(colors, percentages)):
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0]), int(color[1]), int(color[2])
        )
        color_boxes[i].config(bg=hex_color)
        color_labels[i].config(
            text=f"{pct:.1f}%\nRGB({color[0]},{color[1]},{color[2]})"
        )
       

window = tk.Tk()
window.title("Smart Image Analyzer")
window.geometry("700x600")
window.config(bg="#1e1e2e")

tk.Label(
    window, text="Smart Image Analyzer",
    font=("Arial", 20, "bold"),
    bg="#1e1e2e", fg="white"
).pack(pady=15)

tk.Button(
    window, text="Upload Image",
    font=("Arial", 13),
    bg="#7c3aed", fg="white",
    padx=20, pady=8,
    command=upload_and_analyze
).pack(pady=10)

image_label = tk.Label(window, bg="#1e1e2e")
image_label.pack()

result_label = tk.Label(
    window, text="",
    font=("Arial", 14, "bold"),
    bg="#1e1e2e", fg="#a3e635"
)
result_label.pack(pady=10)

tk.Label(
    window, text="Dominant Colors:",
    font=("Arial", 12),
    bg="#1e1e2e", fg="white"
).pack()

colors_frame = tk.Frame(window, bg="#1e1e2e")
colors_frame.pack(pady=5)

color_boxes = []
color_labels = []
for i in range(5):
    box = tk.Label(
        colors_frame, width=8, height=3,
        bg="#333", relief="ridge"
    )
    box.grid(row=0, column=i, padx=8)
    color_boxes.append(box)
    lbl = tk.Label(
        colors_frame, text="",
        font=("Arial", 8),
        bg="#1e1e2e", fg="white"
    )
    lbl.grid(row=1, column=i, padx=8)
    color_labels.append(lbl)

window.mainloop()
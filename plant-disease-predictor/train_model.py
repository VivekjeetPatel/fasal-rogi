import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# CELL 1 – Download Dataset via Kaggle API
# ─────────────────────────────────────────────
# Place your kaggle.json in ~/.kaggle/ (Linux/macOS)
# or C:\Users\<you>\.kaggle\kaggle.json (Windows)
# before running this cell.

os.system("kaggle datasets download -d emmarex/plantdisease --unzip -p ./data")
print("✅ Dataset downloaded and extracted to ./data/")

# ─────────────────────────────────────────────
# CELL 2 – Data Preprocessing
# ─────────────────────────────────────────────

# The PlantVillage dataset unpacks to:
#   ./data/PlantVillage/   (38 class sub-folders)
DATASET_DIR = "./data/PlantVillage"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"✅ Found {NUM_CLASSES} classes")
print(f"   Train samples : {train_generator.samples}")
print(f"   Val   samples : {val_generator.samples}")

# ─────────────────────────────────────────────
# CELL 3 – CNN Architecture
# ─────────────────────────────────────────────

model = keras.Sequential(
    [
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ],
    name="PlantDiseaseCNN",
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─────────────────────────────────────────────
# CELL 4 – Training
# ─────────────────────────────────────────────

EPOCHS = 10

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
)

# ─────────────────────────────────────────────
# CELL 5 – Save Model & Class Indices
# ─────────────────────────────────────────────

os.makedirs("app/trained_model", exist_ok=True)

# Save the Keras model
model.save("app/trained_model/plant_disease_model.h5")
print("✅ Model saved → app/trained_model/plant_disease_model.h5")

# Invert the class_indices dict  { class_name: int } → { str(int): class_name }
class_indices = train_generator.class_indices
inverted_indices = {str(v): k for k, v in class_indices.items()}

with open("app/class_indices.json", "w") as f:
    json.dump(inverted_indices, f, indent=4)

print("✅ Class indices saved → app/class_indices.json")
print("\nSample entries:", dict(list(inverted_indices.items())[:5]))

"""
PlantCare AI - Training Script
Transfer Learning with MobileNetV2
Dataset: 15 classes matching your exact folder names
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
LABELS_PATH   = os.path.join(BASE_DIR, "labels.json")
MODEL_SAVE    = os.path.join(MODEL_DIR, "plant_model.h5")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 5
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)

SELECTED_CLASSES = [
    "Tomato_healthy",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Target_Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Leaf_Mold",
    "Tomato_Late_blight",
    "Tomato_Early_blight",
    "Tomato_Bacterial_spot",
    "Potato___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot",
]
# ─────────────────────────────────────────
# STEP 1 — VERIFY DATASET FOLDERS
# ─────────────────────────────────────────
def verify_dataset():
    print("\n🔍  Verifying dataset folders...\n")
    missing = []
    for cls in SELECTED_CLASSES:
        path = os.path.join(DATASET_DIR, cls)
        if not os.path.isdir(path):
            missing.append(cls)
        else:
            count = len(os.listdir(path))
            print(f"  ✅  {cls}  ({count} images)")

    if missing:
        print("\n❌  Missing folders:")
        for m in missing:
            print(f"    → {m}")
        raise FileNotFoundError("Some dataset folders are missing. Check names carefully.")

    print(f"\n✅  All {len(SELECTED_CLASSES)} classes verified!\n")


# ─────────────────────────────────────────
# STEP 2 — DATA GENERATORS
# ─────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        horizontal_flip=True,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2,
    )

    eval_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=SELECTED_CLASSES,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = eval_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=SELECTED_CLASSES,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    print(f"📊  Training samples  : {train_gen.samples}")
    print(f"📊  Validation samples: {val_gen.samples}")
    print(f"📊  Classes           : {train_gen.num_classes}\n")

    return train_gen, val_gen


# ─────────────────────────────────────────
# STEP 3 — BUILD MODEL
# ─────────────────────────────────────────
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # Freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"✅  Model built — {model.count_params():,} total params\n")
    return model, base_model


# ─────────────────────────────────────────
# STEP 4 — PHASE 1: TRAIN HEAD
# ─────────────────────────────────────────
def train_phase1(model, train_gen, val_gen):
    print("=" * 50)
    print("🚀  PHASE 1 — Training custom head (base frozen)")
    print("=" * 50 + "\n")

    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────
# STEP 5 — PHASE 2: FINE-TUNE
# ─────────────────────────────────────────
def train_phase2(model, base_model, train_gen, val_gen):
    print("\n" + "=" * 50)
    print("🔬  PHASE 2 — Fine-tuning top 30 layers")
    print("=" * 50 + "\n")

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]

    history2 = model.fit(
        train_gen,
        epochs=5,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )
    return history2


# ─────────────────────────────────────────
# STEP 6 — SAVE LABELS
# ─────────────────────────────────────────
def save_labels(class_indices):
    labels = {str(v): k for k, v in class_indices.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n✅  Labels saved → {LABELS_PATH}")


# ─────────────────────────────────────────
# STEP 7 — PRINT SUMMARY
# ─────────────────────────────────────────
def print_results(h1, h2=None):
    all_val = h1.history["val_accuracy"] + (h2.history["val_accuracy"] if h2 else [])
    last = (h2 or h1).history
    print("\n" + "=" * 50)
    print("📈  TRAINING COMPLETE")
    print("=" * 50)
    print(f"   Final Train Accuracy : {last['accuracy'][-1]*100:.2f}%")
    print(f"   Final Val   Accuracy : {last['val_accuracy'][-1]*100:.2f}%")
    print(f"   Best  Val   Accuracy : {max(all_val)*100:.2f}%")
    print(f"   Model saved → {MODEL_SAVE}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n🌿  PlantCare AI — Training Pipeline")
    print(f"    Classes : {len(SELECTED_CLASSES)}")
    print(f"    Dataset : {DATASET_DIR}\n")

    verify_dataset()
    train_gen, val_gen = build_generators()
    model, base_model  = build_model(num_classes=len(SELECTED_CLASSES))
    h1 = train_phase1(model, train_gen, val_gen)
    h2 = train_phase2(model, base_model, train_gen, val_gen)
    save_labels(train_gen.class_indices)
    print_results(h1, h2)

"""
PlantCare AI - Flask Backend
Matches exact folder names from user's dataset (15 classes)
POST /predict  →  { disease, confidence, remedy, category, severity }
GET  /stats    →  dashboard statistics
"""

import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "plant_model.h5")
LABELS_PATH  = os.path.join(BASE_DIR, "labels.json")
UPLOAD_DIR   = os.path.join(BASE_DIR, "static", "uploads")
IMG_SIZE     = (224, 224)
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────
# REMEDIES — exact folder name keys
# ─────────────────────────────────────────
REMEDIES = {
    "Tomato_healthy": {
        "display":  "Tomato — Healthy",
        "remedy":   "Your tomato plant is healthy! Maintain consistent watering, stake for support, and scout weekly for early disease signs.",
        "severity": "None",
        "category": "Healthy",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display":  "Tomato — Mosaic Virus",
        "remedy":   "No chemical cure. Remove and destroy infected plants immediately. Disinfect tools with 10% bleach. Control aphid vectors with insecticidal soap. Wash hands before handling plants.",
        "severity": "High",
        "category": "Viral",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display":  "Tomato — Yellow Leaf Curl Virus",
        "remedy":   "No cure — remove infected plants immediately. Control whitefly vectors with yellow sticky traps and neem oil. Use virus-resistant varieties next season. Apply reflective mulch to repel whiteflies.",
        "severity": "Critical",
        "category": "Viral",
    },
    "Tomato__Target_Spot": {
        "display":  "Tomato — Target Spot",
        "remedy":   "Apply fungicides containing azoxystrobin or chlorothalonil. Remove infected leaves. Improve plant spacing for air circulation. Use drip watering instead of overhead irrigation.",
        "severity": "Moderate",
        "category": "Fungal",
    },
    "Tomato_Spider_mites_Two_spotted_spid": {
        "display":  "Tomato — Spider Mites (Two-Spotted)",
        "remedy":   "Apply miticides or insecticidal soap spray. Increase humidity around plants. Introduce predatory mites as biological control. Remove heavily infested leaves immediately.",
        "severity": "Moderate",
        "category": "Pest",
    },
    "Tomato_Septoria_leaf_spot": {
        "display":  "Tomato — Septoria Leaf Spot",
        "remedy":   "Remove infected lower leaves. Apply chlorothalonil or mancozeb every 7–10 days. Mulch around base to prevent soil splash. Practice 2-year crop rotation.",
        "severity": "Moderate",
        "category": "Fungal",
    },
    "Tomato_Leaf_Mold": {
        "display":  "Tomato — Leaf Mold",
        "remedy":   "Improve ventilation. Remove infected leaves. Apply copper-based fungicide or chlorothalonil. Keep humidity below 85%. Avoid overhead watering.",
        "severity": "Moderate",
        "category": "Fungal",
    },
    "Tomato_Late_blight": {
        "display":  "Tomato — Late Blight",
        "remedy":   "Act immediately — highly contagious! Remove and bag infected plants. Apply mancozeb or chlorothalonil. Avoid overhead watering. Destroy ALL plant debris.",
        "severity": "Critical",
        "category": "Oomycete",
    },
    "Tomato_Early_blight": {
        "display":  "Tomato — Early Blight",
        "remedy":   "Remove infected lower leaves. Apply chlorothalonil every 7 days. Mulch around base. Stake plants to improve airflow. Rotate crops annually.",
        "severity": "Moderate",
        "category": "Fungal",
    },
    "Tomato_Bacterial_spot": {
        "display":  "Tomato — Bacterial Spot",
        "remedy":   "Apply copper hydroxide bactericide. Use certified disease-free seeds. Avoid overhead irrigation. Remove crop debris after harvest. Rotate crops 2+ years.",
        "severity": "Moderate",
        "category": "Bacterial",
    },
    "Potato__healthy": {
        "display":  "Potato — Healthy",
        "remedy":   "Your potato plant is healthy! Hill soil around stems, maintain consistent moisture, and check weekly for late blight in humid weather.",
        "severity": "None",
        "category": "Healthy",
    },
    "Potato__Late_blight": {
        "display":  "Potato — Late Blight",
        "remedy":   "URGENT — extremely contagious! Remove and bag infected plants immediately. Apply metalaxyl or cymoxanil fungicides. Avoid wet field work. Destroy ALL debris.",
        "severity": "Critical",
        "category": "Oomycete",
    },
    "Potato__Early_blight": {
        "display":  "Potato — Early Blight",
        "remedy":   "Remove infected lower leaves. Apply chlorothalonil or mancozeb. Ensure proper plant spacing. Rotate crops every year. Avoid overhead watering.",
        "severity": "Moderate",
        "category": "Fungal",
    },
    "Pepper__bell__healthy": {
        "display":  "Pepper (Bell) — Healthy",
        "remedy":   "Your pepper plant is healthy! Ensure consistent watering, 6–8 hrs sunlight, and stake tall plants. Feed with balanced fertilizer every 2 weeks.",
        "severity": "None",
        "category": "Healthy",
    },
    "Pepper__bell___Bacterial_spot": {
        "display":  "Pepper (Bell) — Bacterial Spot",
        "remedy":   "Apply copper hydroxide sprays every 5–7 days. Avoid working with wet plants. Use certified disease-free seeds. Rotate crops 2+ years.",
        "severity": "Moderate",
        "category": "Bacterial",
    },
}

DEFAULT_REMEDY = {
    "display":  "Unknown",
    "remedy":   "Consult a local agricultural expert for diagnosis and treatment.",
    "severity": "Unknown",
    "category": "Unknown",
}

SEVERITY_COLOR = {
    "None":     "#22c55e",
    "Moderate": "#f59e0b",
    "High":     "#f97316",
    "Critical": "#ef4444",
    "Unknown":  "#6b7280",
}


# ─────────────────────────────────────────
# LOAD MODEL & LABELS
# ─────────────────────────────────────────
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model not found at %s — run train.py first.", MODEL_PATH)
        return None, {}
    if not os.path.exists(LABELS_PATH):
        logger.warning("labels.json not found at %s.", LABELS_PATH)
        return None, {}
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    logger.info("Model loaded — %d classes", len(labels))
    return model, labels


MODEL, LABELS = load_model_and_labels()

# ─────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def preprocess_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stats")
def stats():
    if not LABELS:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    category_counts = {}
    severity_counts = {}
    for label in LABELS.values():
        info = REMEDIES.get(label, DEFAULT_REMEDY)
        cat  = info["category"]
        sev  = info["severity"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    class_list = []
    for label in LABELS.values():
        info = REMEDIES.get(label, DEFAULT_REMEDY)
        class_list.append({
            "raw":      label,
            "display":  info["display"],
            "category": info["category"],
            "severity": info["severity"],
        })

    return jsonify({
        "total_classes":      len(LABELS),
        "model_loaded":       MODEL is not None,
        "category_breakdown": category_counts,
        "severity_breakdown": severity_counts,
        "classes":            class_list,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided. Use key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTS)}"}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    logger.info("Uploaded: %s", save_path)

    try:
        img_array   = preprocess_image(save_path)
        predictions = MODEL.predict(img_array, verbose=0)[0]
        class_idx   = int(np.argmax(predictions))
        confidence  = float(np.max(predictions)) * 100

        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                "disease":    REMEDIES.get(LABELS.get(str(i), ""), DEFAULT_REMEDY)["display"],
                "confidence": f"{predictions[i]*100:.1f}%",
            }
            for i in top3_idx
        ]

        disease_raw = LABELS.get(str(class_idx), "Unknown")
        info        = REMEDIES.get(disease_raw, DEFAULT_REMEDY)

        logger.info("Predicted: %s (%.1f%%)", info["display"], confidence)

        return jsonify({
            "disease":        info["display"],
            "disease_raw":    disease_raw,
            "confidence":     f"{confidence:.1f}%",
            "confidence_num": round(confidence, 1),
            "remedy":         info["remedy"],
            "category":       info["category"],
            "severity":       info["severity"],
            "severity_color": SEVERITY_COLOR.get(info["severity"], "#6b7280"),
            "top3":           top3,
        })

    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        return jsonify({"error": "Prediction failed. Please try again."}), 500


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("PlantCare AI running on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
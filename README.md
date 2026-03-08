# PlantCare AI 🌿

AI-powered plant disease detection system using TensorFlow and Flask.

## Features
- Upload plant leaf image
- AI predicts disease with confidence score
- 15 plant disease classes (Tomato, Potato, Pepper)
- Detailed treatment remedies for each disease
- Web interface using Flask
- REST API endpoints

## Tech Stack
- Python
- TensorFlow (MobileNetV2)
- Flask
- OpenCV
- Pillow

## Supported Plants
- Tomato (10 conditions)
- Potato (3 conditions)
- Pepper/Bell Pepper (2 conditions)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model: `python train.py`
2. Run the Flask app: `python app.py`
3. Open browser and access at: `http://localhost:5000`
4. Upload a plant leaf image to get disease prediction
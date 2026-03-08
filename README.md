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

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/Shrau26/PlantCare-AI.git
cd PlantCare-AI

# Install dependencies
pip install -r requirements.txt
```

## API Endpoints

### POST /predict
Upload a plant leaf image to get disease prediction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file (png, jpg, jpeg, webp)

**Response:**
```json
{
  "disease": "Tomato — Early Blight",
  "confidence": "95.2%",
  "remedy": "Remove infected lower leaves...",
  "category": "Fungal",
  "severity": "Moderate"
}
```

### GET /stats
Get dashboard statistics about the model.

## Usage
1. Train the model: `python train.py`
2. Run the Flask app: `python app.py`
3. Open browser and access at: `http://localhost:5000`
4. Upload a plant leaf image to get disease prediction
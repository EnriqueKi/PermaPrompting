# Iris/Soil Chromatography Webapp

A full-stack web application for analyzing circular chromatography (or human iris) images and provide (artistic) interpretations.

⚠️ This is an experimental mostly vibe-coded hackathon project ⚠️

We do apply some computer vision methods for detecting visual features (center, regions, etc.) and turn those numeric results into natural language interpretations with an LLM. There is more stuff happening.

The Frontend is written in vanilla Javascript and the backend is just a small Python API built with Flask.

⚠️ This is also not an "AI Agent" in the 2024-will-be-the-year-of-AI-agents sense. There is no _tool use_ and no _reasoning_. We just call the Claude API with some prompts that we hardcoded somewhere.

## 📁 Project Structure

```
chromatography/
├── backend/                     # Backend API server
│   ├── api.py                  # Flask API server
│   ├── chromatography_analyzer.py  # Core analysis logic
│   ├── requirements.txt        # Python dependencies
│   ├── api_client_example.py   # Python client example
│   ├── index.py               # Alternative entry point
│   ├── README_API.md          # API documentation
│   └── uploads/               # Temporary file uploads
├── frontend/                   # Frontend web interface
│   ├── index.html             # Main web interface
│   ├── styles.css             # Styling
│   └── script.js              # Frontend JavaScript
├── *.jpg, *.jpeg              # Sample chromatography images
└── analysis_results_*.json    # Sample analysis results
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

Use the automated startup script that handles conda environment activation:

```bash
./start.sh
```

This will:

- Activate the `summerschoolenv` conda environment (or create it if needed)
- Install dependencies
- Start both backend and frontend servers
- Open the application in your browser

### Option 2: Manual Setup

#### Backend Setup

1. **Set up conda environment** (if using conda):

   ```bash
   # One-time setup
   ./setup_conda_env.sh

   # Or manually:
   conda activate summerschoolenv
   ```

2. **Navigate to the backend directory**:

   ```bash
   cd backend
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the API server**:

   ```bash
   python api.py
   # Or use the startup script:
   ./start_server.sh
   ```

   The API will be available at `http://localhost:8080`

#### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Open the web interface:

   - **Option 1**: Open `index.html` directly in your browser
   - **Option 2**: Use a simple HTTP server:

     ```bash
     # Python 3
     python -m http.server 3000

     # Node.js (if you have it)
     npx serve .
     ```

3. Access the web interface at `http://localhost:3000` (if using a server) or by opening the HTML file directly.

## 🔬 How to Use

### Web Interface

1. **Upload Image**:

   - Click the upload area or drag and drop your chromatography image
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF
   - Maximum file size: 16MB

2. **Set Parameters**:

   - **Number of Regions**: 1-10 (default: 5)
   - **Segmentation Method**: Choose from 4 available methods
   - **Paper Diameter**: In centimeters (default: 12.5)

3. **Analyze**: Click "Analyze Chromatogram" to process your image

4. **View Results**:
   - Analysis summary with total zones and area
   - Detailed zone analysis with colors and measurements
   - Radial analysis (if available)

### API Usage

The backend provides a RESTful API for programmatic access:

#### Endpoints

- `GET /health` - Check API status
- `POST /analyze` - Analyze image via file upload
- `POST /analyze_base64` - Analyze image via base64 data

#### Example with curl:

```bash
curl -X POST \
  -F "file=@your_image.jpg" \
  -F "n_regions=5" \
  -F "segmentation_method=fcm_approx" \
  -F "paper_diameter_cm=12.5" \
  http://localhost:8080/analyze
```

#### Example with Python:

```python
import requests

# File upload method
with open('chromatogram.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'n_regions': 5,
        'segmentation_method': 'fcm_approx',
        'paper_diameter_cm': 12.5
    }
    response = requests.post('http://localhost:8080/analyze', files=files, data=data)
    results = response.json()

print(f"Total zones: {results['summary']['total_zones']}")
```

## 🛠️ Configuration

### Backend Configuration

- **Port**: Default 8080 (change in `api.py`)
- **Upload folder**: `uploads/` (auto-created)
- **Max file size**: 16MB
- **Allowed formats**: PNG, JPG, JPEG, BMP, TIFF

### Frontend Configuration

- **API URL**: `http://localhost:8080` (change in `script.js`)
- **File size limit**: 16MB (matches backend)

## 📊 Analysis Features

### Zone Detection

- Automatic detection of colored zones
- Width measurements in centimeters
- Color analysis (RGB/Grayscale)
- Zone naming and categorization

### Radial Analysis

- Channel development analysis
- Spike development detection
- Statistical measurements
- Density calculations

### Segmentation Methods

- **fcm_approx**: Fuzzy C-Means approximation (default)
- **radial_guided**: Radial-guided segmentation
- **radial_kmeans**: Pure radial K-means
- **gmm**: Gaussian Mixture Model

## 🔧 Development

### Backend Development

```bash
cd backend
python api.py  # Runs in debug mode
```

### Frontend Development

For development with live reload, you can use any HTTP server:

```bash
cd frontend

# With Python
python -m http.server 3000

# With Node.js live-server
npx live-server --port=3000

# With PHP
php -S localhost:3000
```

### Adding CORS Support

If you encounter CORS issues during development, install and configure Flask-CORS:

```bash
pip install flask-cors
```

Add to `api.py`:

```python
from flask_cors import CORS
CORS(app)
```

## 📝 API Response Format

```json
{
  "success": true,
  "center": [x, y],
  "summary": {
    "total_zones": 5,
    "total_area_cm2": 123.45
  },
  "zones": {
    "zone_1": {
      "zone_full_name": "Zone 1",
      "width_cm": 1.234,
      "color": {"red": 255, "green": 0, "blue": 0}
    }
  },
  "radial_analysis": {
    "channel_development": {...},
    "spike_development": {...}
  },
  "analysis_parameters": {
    "n_regions": 5,
    "segmentation_method": "fcm_approx",
    "paper_diameter_cm": 12.5
  }
}
```

## 🎯 Sample Images

The repository includes sample chromatography images for testing:

- `soilchromatographyC2_back_smaller_large.jpeg`
- `soilchromatographyC2_back_smaller.jpg`
- `synthetic_chromatogram.jpg`
- `test_chromatogram.jpg`

## ⚠️ Troubleshooting

### Common Issues

1. **"Cannot connect to backend"**:

   - Ensure the backend server is running on port 8080
   - Check if the API URL in `script.js` matches your backend

2. **File upload fails**:

   - Check file size (max 16MB)
   - Verify file format is supported
   - Ensure uploads directory exists and is writable

3. **Analysis fails**:
   - Check image is valid
   - Verify parameters are within valid ranges
   - Check backend logs for detailed error messages

### CORS Issues

If using the frontend from a different domain/port:

1. Install Flask-CORS: `pip install flask-cors`
2. Add CORS support to `api.py`

## 📄 License

This project is open source. See the API documentation for more details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both frontend and backend
5. Submit a pull request

## 📚 Dependencies

### Backend

- Flask 3.0.0
- OpenCV 4.8.1.78
- NumPy 1.26.2
- Scikit-image 0.22.0
- Scikit-learn 1.3.2
- SciPy 1.11.4
- Requests 2.31.0

### Frontend

- Pure HTML/CSS/JavaScript (no external dependencies)
- Modern browser with File API support
- Fetch API support

## 🐍 Conda Environment

This project is designed to work with the `summerschoolenv` conda environment. The startup scripts will automatically:

1. **Detect and activate** the conda environment
2. **Create the environment** if it doesn't exist
3. **Install dependencies** in the correct environment

### Manual Conda Setup

```bash
# Create environment
conda create -n summerschoolenv python=3.9

# Activate environment
conda activate summerschoolenv

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Environment Management

```bash
# Check if environment exists
conda env list | grep summerschoolenv

# Remove environment (if needed)
conda env remove -n summerschoolenv

# Export environment
conda env export -n summerschoolenv > environment.yml
```

---

For detailed API documentation, see `backend/README_API.md`.

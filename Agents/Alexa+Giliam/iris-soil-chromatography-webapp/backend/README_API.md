# Chromatography Analysis API

A simple Flask API for analyzing circular chromatography images and extracting zone features.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:5000`

### 3. Test the API

```bash
python api_client_example.py
```

## 📖 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### `GET /` - API Documentation

Returns API documentation and available endpoints.

#### `GET /health` - Health Check

Returns API status and version information.

#### `POST /analyze` - Analyze Chromatogram (File Upload)

Upload an image file and get analysis results.

**Parameters:**

- `file` (required): Image file (PNG, JPG, JPEG, BMP, TIFF)
- `n_regions` (optional): Number of regions to detect (1-10, default: 5)
- `segmentation_method` (optional): Segmentation method (default: fcm_approx)
  - `radial_guided`: Radial-guided segmentation
  - `radial_kmeans`: Pure radial K-means
  - `gmm`: Gaussian Mixture Model
  - `fcm_approx`: Fuzzy C-Means approximation
- `paper_diameter_cm` (optional): Paper diameter in cm (default: 12.5)

**Example with curl:**

```bash
curl -X POST \
  -F "file=@your_chromatogram.jpg" \
  -F "n_regions=5" \
  -F "segmentation_method=fcm_approx" \
  -F "paper_diameter_cm=12.5" \
  http://localhost:5000/analyze
```

#### `POST /analyze_base64` - Analyze Chromatogram (Base64)

Send base64-encoded image data and get analysis results.

**JSON Payload:**

```json
{
  "image_data": "base64_encoded_image_data",
  "n_regions": 5,
  "segmentation_method": "fcm_approx",
  "paper_diameter_cm": 12.5
}
```

**Example with curl:**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "iVBORw0KGgoAAAANSUhEUgAA...",
    "n_regions": 5,
    "segmentation_method": "fcm_approx",
    "paper_diameter_cm": 12.5
  }' \
  http://localhost:5000/analyze_base64
```

## 📊 Response Format

### Success Response

```json
{
  "success": true,
  "center": [250, 250],
  "zones": {
    "CZ": {
      "zone_name": "CZ",
      "zone_full_name": "Central zone",
      "width_cm": 0.123,
      "area_cm2": 1.45,
      "mean_radius_cm": 2.1,
      "thickness_cm": 0.8,
      "angular_coverage": 360.0,
      "mean_intensity": 145.2,
      "contrast": 23.5,
      "color": {
        "red": 145,
        "green": 132,
        "blue": 98
      }
    },
    "MZ": {
      "zone_name": "MZ",
      "zone_full_name": "Median zone",
      "width_cm": 0.087,
      "area_cm2": 2.13,
      "mean_radius_cm": 3.2,
      "thickness_cm": 1.1,
      "angular_coverage": 360.0,
      "mean_intensity": 178.1,
      "contrast": 18.7,
      "color": {
        "red": 178,
        "green": 165,
        "blue": 142
      }
    },
    "OZ": {
      "zone_name": "OZ",
      "zone_full_name": "Outer zone",
      "width_cm": 0.156,
      "area_cm2": 3.78,
      "mean_radius_cm": 4.5,
      "thickness_cm": 1.4,
      "angular_coverage": 360.0,
      "mean_intensity": 201.3,
      "contrast": 15.2,
      "color": {
        "red": 201,
        "green": 188,
        "blue": 167
      }
    }
  },
  "summary": {
    "total_zones": 5,
    "total_area_cm2": 12.34,
    "average_radius_cm": 3.8,
    "paper_diameter_cm": 12.5,
    "resolution_pixels_per_cm": 40.0
  },
  "radial_analysis": {
    "channel_development": {
      "total_channels": 23,
      "avg_channel_length_cm": 1.45,
      "channel_density": 0.234,
      "avg_continuity": 0.78
    },
    "spike_development": {
      "total_spikes": 12,
      "spike_density": 5.67,
      "avg_spike_intensity": 45.2
    },
    "radial_uniformity": {
      "radial_consistency": 0.85,
      "radial_smoothness": 0.92,
      "development_extent_cm": 4.2
    },
    "directional_patterns": {
      "intensity_asymmetry": 0.12,
      "dominant_direction_degrees": 87.5,
      "symmetry_score": 0.88
    }
  },
  "analysis_parameters": {
    "n_regions": 5,
    "segmentation_method": "fcm_approx",
    "paper_diameter_cm": 12.5,
    "filename": "chromatogram.jpg"
  }
}
```

### Error Response

```json
{
  "success": false,
  "error": "Invalid file format",
  "message": "Allowed formats: png, jpg, jpeg, bmp, tiff, tif"
}
```

## 🔬 Zone Analysis

The API automatically identifies the three primary chromatographic zones:

- **CZ (Central zone)**: Innermost region around the application point
- **MZ (Median zone)**: Middle separation region
- **OZ (Outer zone)**: Outer separation region

Each zone includes:

- **Geometric features**: Width, area, radius, thickness, angular coverage
- **Intensity features**: Mean intensity and contrast
- **Color information**: RGB values for each zone

## 🔍 Radial Features

Advanced radial analysis includes:

- **Channel Development**: Analysis of radial migration paths
- **Spike Development**: Detection of localized features or contamination
- **Radial Uniformity**: Assessment of development consistency
- **Directional Patterns**: Analysis of asymmetries and preferred directions

## 📁 File Support

**Supported formats:**

- PNG
- JPG/JPEG
- BMP
- TIFF/TIF

**Maximum file size:** 16MB

## 🚀 Python Client Example

```python
import requests

# Analyze chromatogram with file upload
with open('chromatogram.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'n_regions': 5,
        'segmentation_method': 'fcm_approx',
        'paper_diameter_cm': 12.5
    }
    response = requests.post('http://localhost:5000/analyze', files=files, data=data)
    results = response.json()

# Print zone analysis
for zone_key, zone_data in results['zones'].items():
    print(f"{zone_key}: Width={zone_data['width_cm']:.3f}cm")
```

## 🛠️ Development

### Running in Development Mode

```bash
python api.py
```

### Running in Production

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

## 📋 Error Codes

- `400` - Bad Request (invalid parameters, missing file, etc.)
- `404` - Endpoint not found
- `405` - Method not allowed
- `413` - File too large (>16MB)
- `500` - Server error (analysis failed, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the provided examples
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

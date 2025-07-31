#!/usr/bin/env python3
"""
Simple Flask API for Chromatography Analysis
Provides endpoints to upload chromatography images and get analysis results as JSON
"""

import os
import json
import base64
from io import BytesIO
from typing import Dict, Any
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from chromatography_analyzer import CircularChromatographyAnalyzer
from claude_service import claude_service

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for frontend
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize analyzer
analyzer = CircularChromatographyAnalyzer()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_visualization_base64(analyzer, results):
    """Generate visualization as base64 encoded image"""
    try:
        # Clear any existing plots
        plt.clf()
        
        # Use the analyzer's built-in visualization method
        analyzer.visualize_results(results)
        
        # Save the current figure to a BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Clean up
        plt.close('all')
        buffer.close()
        
        return image_base64
        
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        # Clean up in case of error
        plt.close('all')
        return None

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def prepare_analysis_results(results: Dict, analyzer=None) -> Dict[str, Any]:
    """Prepare analysis results for JSON serialization"""
    # Extract key information without large arrays
    json_results = {
        'success': True,
        'center': convert_numpy_types(results['center']),
        'zones': {},
        'summary': {},
        'radial_features': convert_numpy_types(results.get('radial_features', {}))
    }
    
    # Generate visualization if analyzer is provided
    if analyzer:
        visualization_base64 = generate_visualization_base64(analyzer, results)
        if visualization_base64:
            json_results['visualization'] = f"data:image/png;base64,{visualization_base64}"
    
    # Process zone features
    zone_order = ['CZ', 'MZ', 'OZ']
    total_area = 0
    zone_count = 0
    
    for zone_key in zone_order:
        if zone_key in results['features']:
            features = results['features'][zone_key]
            zone_data = {
                'zone_name': features.get('zone_name', zone_key),
                'zone_full_name': features.get('zone_full_name', f'Zone {zone_key}'),
                'width_cm': convert_numpy_types(features['width_cm']),
                'area_cm2': convert_numpy_types(features['area_cm2']),
                'mean_radius_cm': convert_numpy_types(features['mean_radius_cm']),
                'thickness_cm': convert_numpy_types(features['thickness_cm']),
                'angular_coverage': convert_numpy_types(features['angular_coverage']),
                'mean_intensity': convert_numpy_types(features['mean_intensity']),
                'contrast': convert_numpy_types(features['contrast']),
                'color': convert_numpy_types(features['color'])
            }
            json_results['zones'][zone_key] = zone_data
            total_area += features['area_cm2']
            zone_count += 1
    
    # Add additional regions if they exist
    additional_regions = {}
    for region_key, features in results['features'].items():
        if region_key not in zone_order:
            region_data = {
                'zone_full_name': features.get('zone_full_name', region_key),
                'width_cm': convert_numpy_types(features['width_cm']),
                'area_cm2': convert_numpy_types(features['area_cm2']),
                'mean_radius_cm': convert_numpy_types(features['mean_radius_cm']),
                'thickness_cm': convert_numpy_types(features['thickness_cm']),
                'angular_coverage': convert_numpy_types(features['angular_coverage']),
                'mean_intensity': convert_numpy_types(features['mean_intensity']),
                'contrast': convert_numpy_types(features['contrast']),
                'color': convert_numpy_types(features['color'])
            }
            additional_regions[region_key] = region_data
            total_area += features['area_cm2']
            zone_count += 1
    
    if additional_regions:
        json_results['additional_regions'] = additional_regions
    
    # Summary statistics
    json_results['summary'] = {
        'total_zones': zone_count,
        'total_area_cm2': convert_numpy_types(total_area),
        'average_radius_cm': convert_numpy_types(
            sum(f['mean_radius_cm'] for f in results['features'].values()) / len(results['features'])
        ),
        'paper_diameter_cm': analyzer.paper_diameter_cm,
        'resolution_pixels_per_cm': convert_numpy_types(analyzer.resolution) if analyzer.resolution else None
    }
    
    # Add radial feature summary if available
    if 'radial_features' in results:
        rf = results['radial_features']
        radial_summary = {}
        
        if 'channel_development' in rf:
            ch = rf['channel_development']
            radial_summary['channel_development'] = {
                'total_channels': convert_numpy_types(ch['total_channels']),
                'avg_channel_length_cm': convert_numpy_types(ch['avg_channel_length_cm']),
                'channel_density': convert_numpy_types(ch['channel_density']),
                'avg_continuity': convert_numpy_types(ch['avg_continuity'])
            }
        
        if 'spike_development' in rf:
            sp = rf['spike_development']
            radial_summary['spike_development'] = {
                'total_spikes': convert_numpy_types(sp['total_spikes']),
                'spike_density': convert_numpy_types(sp['spike_density']),
                'avg_spike_intensity': convert_numpy_types(sp['avg_spike_intensity'])
            }
        
        if 'radial_uniformity' in rf:
            ru = rf['radial_uniformity']
            radial_summary['radial_uniformity'] = {
                'radial_consistency': convert_numpy_types(ru['radial_consistency']),
                'radial_smoothness': convert_numpy_types(ru['radial_smoothness']),
                'development_extent_cm': convert_numpy_types(ru['development_extent_cm'])
            }
        
        if 'directional_patterns' in rf:
            dp = rf['directional_patterns']
            radial_summary['directional_patterns'] = {
                'intensity_asymmetry': convert_numpy_types(dp['intensity_asymmetry']),
                'dominant_direction_degrees': convert_numpy_types(dp['dominant_direction_degrees']),
                'symmetry_score': convert_numpy_types(dp['symmetry_score'])
            }
        
        json_results['radial_analysis'] = radial_summary
    
    return json_results

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    docs = {
        'api': 'Chromatography Analysis API',
        'version': '1.0',
        'endpoints': {
            'POST /analyze': {
                'description': 'Analyze a chromatography image',
                'parameters': {
                    'file': 'Image file (PNG, JPG, JPEG, BMP, TIFF)',
                    'n_regions': 'Number of regions to detect (optional, default: 5)',
                    'segmentation_method': 'Segmentation method (optional, default: fcm_approx)',
                    'paper_diameter_cm': 'Paper diameter in cm (optional, default: 12.5)'
                },
                'returns': 'JSON with analysis results'
            },
            'POST /analyze_base64': {
                'description': 'Analyze a chromatography image from base64 data',
                'parameters': {
                    'image_data': 'Base64 encoded image data',
                    'n_regions': 'Number of regions to detect (optional, default: 5)',
                    'segmentation_method': 'Segmentation method (optional, default: fcm_approx)',
                    'paper_diameter_cm': 'Paper diameter in cm (optional, default: 12.5)'
                },
                'returns': 'JSON with analysis results'
            },
            'GET /health': {
                'description': 'Health check endpoint',
                'returns': 'API status'
            },
            'POST /claude/analyze': {
                'description': 'Generate AI analysis from chromatography results',
                'parameters': {
                    'analysis_results': 'Required: chromatography analysis results in JSON format'
                },
                'returns': 'JSON with Claude AI analysis and insights'
            },
            'POST /claude/feral': {
                'description': 'Generate poetic soil interpretation from Claude analysis results',
                'parameters': {
                    'claude_analysis_data': 'Required: Claude AI analysis results in JSON format'
                },
                'returns': 'JSON with poetic utterances based on Claude analysis insights'
            },
            'GET /claude/status': {
                'description': 'Check Claude AI service availability',
                'returns': 'Claude service status and configuration'
            }
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    }
    return jsonify(docs)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'api': 'Chromatography Analysis API',
        'version': '1.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_chromatogram():
    """Analyze chromatography image endpoint"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select an image file'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file format',
                'message': f'Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get analysis parameters
        n_regions = int(request.form.get('n_regions', 5))
        segmentation_method = request.form.get('segmentation_method', 'fcm_approx')
        paper_diameter_cm = float(request.form.get('paper_diameter_cm', 12.5))
        
        # Validate parameters
        if n_regions < 1 or n_regions > 10:
            return jsonify({
                'success': False,
                'error': 'Invalid n_regions',
                'message': 'n_regions must be between 1 and 10'
            }), 400
        
        if segmentation_method not in ['radial_guided', 'radial_kmeans', 'gmm', 'fcm_approx']:
            return jsonify({
                'success': False,
                'error': 'Invalid segmentation_method',
                'message': 'Must be one of: radial_guided, radial_kmeans, gmm, fcm_approx'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Update analyzer with custom paper diameter
            analyzer.paper_diameter_cm = paper_diameter_cm
            
            # Perform analysis
            results = analyzer.analyze_chromatogram(
                filepath, 
                n_regions=n_regions,
                segmentation_method=segmentation_method
            )
            
            # Prepare JSON response with visualization
            json_results = prepare_analysis_results(results, analyzer)
            json_results['analysis_parameters'] = {
                'n_regions': n_regions,
                'segmentation_method': segmentation_method,
                'paper_diameter_cm': paper_diameter_cm,
                'filename': filename
            }
            
            return jsonify(json_results)
            
        except Exception as analysis_error:
            return jsonify({
                'success': False,
                'error': 'Analysis failed',
                'message': str(analysis_error)
            }), 500
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/analyze_base64', methods=['POST'])
def analyze_chromatogram_base64():
    """Analyze chromatography image from base64 data"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided',
                'message': 'Please provide base64 encoded image data'
            }), 400
        
        # Get analysis parameters
        n_regions = int(data.get('n_regions', 5))
        segmentation_method = data.get('segmentation_method', 'fcm_approx')
        paper_diameter_cm = float(data.get('paper_diameter_cm', 12.5))
        
        # Validate parameters
        if n_regions < 1 or n_regions > 10:
            return jsonify({
                'success': False,
                'error': 'Invalid n_regions',
                'message': 'n_regions must be between 1 and 10'
            }), 400
        
        if segmentation_method not in ['radial_guided', 'radial_kmeans', 'gmm', 'fcm_approx']:
            return jsonify({
                'success': False,
                'error': 'Invalid segmentation_method',
                'message': 'Must be one of: radial_guided, radial_kmeans, gmm, fcm_approx'
            }), 400
        
        try:
            # Decode base64 image
            image_data = data['image_data']
            if image_data.startswith('data:image'):
                # Remove data URL prefix if present
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Invalid image data',
                    'message': 'Could not decode the provided image data'
                }), 400
            
            # Save temporary file
            temp_filename = 'temp_analysis.jpg'
            temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
            cv2.imwrite(temp_filepath, image)
            
            try:
                # Update analyzer with custom paper diameter
                analyzer.paper_diameter_cm = paper_diameter_cm
                
                # Perform analysis
                results = analyzer.analyze_chromatogram(
                    temp_filepath,
                    n_regions=n_regions,
                    segmentation_method=segmentation_method
                )
                
                # Prepare JSON response with visualization
                json_results = prepare_analysis_results(results, analyzer)
                json_results['analysis_parameters'] = {
                    'n_regions': n_regions,
                    'segmentation_method': segmentation_method,
                    'paper_diameter_cm': paper_diameter_cm,
                    'input_method': 'base64'
                }
                
                return jsonify(json_results)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
        
        except Exception as decode_error:
            return jsonify({
                'success': False,
                'error': 'Image decode failed',
                'message': str(decode_error)
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/claude/analyze', methods=['POST'])
def claude_analyze():
    """
    Analyze chromatography results using Claude AI
    Accepts analysis results in JSON format
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'message': 'Send analysis results as JSON data'
            }), 400
        
        analysis_results = request.json.get('analysis_results')
        if not analysis_results:
            return jsonify({
                'success': False,
                'error': 'No analysis results provided',
                'message': 'Please include analysis_results in the JSON request'
            }), 400
        
        # Analyze with Claude
        claude_result = claude_service.analyze_chromatography_results(analysis_results)
        
        return jsonify({
            'success': True,
            'claude_analysis': claude_result,
            'service_available': claude_service.is_available()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Claude analysis failed',
            'message': str(e)
        }), 500

@app.route('/claude/status', methods=['GET'])
def claude_status():
    """
    Get Claude API service status
    """
    return jsonify({
        'service': 'Claude AI Analysis',
        'available': claude_service.is_available(),
        'api_key_configured': claude_service.api_key is not None,
        'endpoints': [
            '/claude/analyze - Generate AI analysis from chromatography results',
            '/claude/feral - Generate poetic soil interpretation',
            '/claude/status - Check Claude service status'
        ]
    })

@app.route('/claude/feral', methods=['POST'])
def claude_feral():
    """
    Generate poetic interpretation using feral prompt based on Claude analysis
    Accepts Claude analysis results in JSON format
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'message': 'Send Claude analysis results as JSON data'
            }), 400
        
        claude_analysis_data = request.json.get('claude_analysis_data')
        if not claude_analysis_data:
            return jsonify({
                'success': False,
                'error': 'No Claude analysis data provided',
                'message': 'Please include claude_analysis_data in the JSON request'
            }), 400
        
        # Generate feral analysis with Claude
        feral_result = claude_service.analyze_chromatography_feral(claude_analysis_data)
        
        # Return the JSON directly (it should have utterances field)
        return jsonify(feral_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Feral analysis failed',
            'message': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Maximum file size is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'Please check the API documentation at /'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed error"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'Please check the API documentation at /'
    }), 405

if __name__ == '__main__':
    print("🚀 Starting Chromatography Analysis API...")
    print("📖 API documentation available at: http://localhost:8080/")
    print("🏥 Health check available at: http://localhost:8080/health")
    print("🔬 Analysis endpoint: POST http://localhost:8080/analyze")
    print("🖼️  Base64 endpoint: POST http://localhost:8080/analyze_base64")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=8080)

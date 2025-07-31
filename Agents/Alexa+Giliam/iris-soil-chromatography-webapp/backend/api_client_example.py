#!/usr/bin/env python3
"""
Example client for the Chromatography Analysis API
Shows how to use both file upload and base64 endpoints
"""

import requests
import json
import base64
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8080"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def analyze_with_file_upload(image_path: str, n_regions: int = 5, 
                           segmentation_method: str = 'fcm_approx',
                           paper_diameter_cm: float = 12.5):
    """Analyze chromatogram using file upload"""
    print(f"📤 Analyzing {image_path} via file upload...")
    
    # Prepare the file and data
    files = {'file': open(image_path, 'rb')}
    data = {
        'n_regions': n_regions,
        'segmentation_method': segmentation_method,
        'paper_diameter_cm': paper_diameter_cm
    }
    
    try:
        # Make the request
        response = requests.post(f"{API_BASE_URL}/analyze", files=files, data=data)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            
            # Print key results
            print(f"\n📊 Results Summary:")
            print(f"   Center: {result['center']}")
            print(f"   Total zones: {result['summary']['total_zones']}")
            print(f"   Total area: {result['summary']['total_area_cm2']:.2f} cm²")
            
            # Print zone details
            print(f"\n🎯 Zone Analysis:")
            for zone_key, zone_data in result['zones'].items():
                color = zone_data['color']
                if 'red' in color:
                    color_str = f"RGB({color['red']:.0f}, {color['green']:.0f}, {color['blue']:.0f})"
                else:
                    color_str = f"Gray({color['gray']:.0f})"
                
                print(f"   {zone_key} ({zone_data['zone_full_name']}): "
                      f"Width={zone_data['width_cm']:.3f}cm, Color={color_str}")
            
            # Print radial analysis if available
            if 'radial_analysis' in result:
                print(f"\n🔍 Radial Analysis:")
                ra = result['radial_analysis']
                
                if 'channel_development' in ra:
                    ch = ra['channel_development']
                    print(f"   📡 Channels: {ch['total_channels']} total, "
                          f"{ch['avg_channel_length_cm']:.2f}cm avg length")
                
                if 'spike_development' in ra:
                    sp = ra['spike_development']
                    print(f"   🔺 Spikes: {sp['total_spikes']} total, "
                          f"density: {sp['spike_density']:.2f}/10k pixels")
            
            return result
        else:
            error_info = response.json()
            print(f"❌ Analysis failed: {error_info.get('error', 'Unknown error')}")
            print(f"   Message: {error_info.get('message', 'No details available')}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None
    finally:
        files['file'].close()

def analyze_with_base64(image_path: str, n_regions: int = 5,
                       segmentation_method: str = 'fcm_approx',
                       paper_diameter_cm: float = 12.5):
    """Analyze chromatogram using base64 encoding"""
    print(f"📤 Analyzing {image_path} via base64...")
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare JSON payload
        payload = {
            'image_data': image_data,
            'n_regions': n_regions,
            'segmentation_method': segmentation_method,
            'paper_diameter_cm': paper_diameter_cm
        }
        
        # Make the request
        response = requests.post(
            f"{API_BASE_URL}/analyze_base64",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            
            # Print simplified results
            print(f"\n📊 Results Summary:")
            print(f"   Total zones: {result['summary']['total_zones']}")
            
            print(f"\n🎯 Zone Colors & Widths:")
            for zone_key, zone_data in result['zones'].items():
                color = zone_data['color']
                if 'red' in color:
                    color_str = f"RGB({color['red']:.0f}, {color['green']:.0f}, {color['blue']:.0f})"
                else:
                    color_str = f"Gray({color['gray']:.0f})"
                
                print(f"   {zone_key}: Width={zone_data['width_cm']:.3f}cm, Color={color_str}")
            
            return result
        else:
            error_info = response.json()
            print(f"❌ Analysis failed: {error_info.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def save_results_to_file(results: dict, output_path: str):
    """Save analysis results to JSON file"""
    if results:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_path}")

def main():
    """Main example function"""
    print("🔬 CHROMATOGRAPHY ANALYSIS API CLIENT")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ API is not responding. Make sure the server is running.")
        return
    
    print("\n" + "=" * 50)
    
    # Example image path (update this to your actual image)
    image_path = "soilchromatographyC2_back_smaller_large.jpeg"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        print("   Please update the image_path variable in this script.")
        return
    
    # Test file upload method
    print("🧪 Testing file upload method...")
    results_upload = analyze_with_file_upload(
        image_path=image_path,
        n_regions=5,
        segmentation_method='fcm_approx',
        paper_diameter_cm=12.5
    )
    
    if results_upload:
        save_results_to_file(results_upload, "analysis_results_upload.json")
    
    print("\n" + "=" * 50)
    
    # Test base64 method
    print("🧪 Testing base64 method...")
    results_base64 = analyze_with_base64(
        image_path=image_path,
        n_regions=5,
        segmentation_method='fcm_approx',
        paper_diameter_cm=12.5
    )
    
    if results_base64:
        save_results_to_file(results_base64, "analysis_results_base64.json")
    
    print("\n" + "=" * 50)
    print("✅ API testing complete!")

if __name__ == "__main__":
    main()

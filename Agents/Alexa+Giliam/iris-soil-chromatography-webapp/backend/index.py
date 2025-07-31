# Basic usage
from chromatography_analyzer import CircularChromatographyAnalyzer
analyzer = CircularChromatographyAnalyzer()

# Get results and visualize immediately
results = analyzer.analyze_chromatogram('soilchromatographyC2_back_smaller_large.jpeg', n_regions=5)

# Print width and color for the primary zones
print("\nZone Analysis:")
print("-" * 30)

# Primary zones (CZ, MZ, OZ) are already named in the analyzer
zone_order = ['CZ', 'MZ', 'OZ']
for zone_key in zone_order:
    if zone_key in results['features']:
        features = results['features'][zone_key]
        width = features['width_cm']
        color = features['color']
        zone_full_name = features['zone_full_name']
        
        if 'red' in color:
            color_str = f"RGB({color['red']:.0f}, {color['green']:.0f}, {color['blue']:.0f})"
        else:
            color_str = f"Gray({color['gray']:.0f})"
        
        print(f"{zone_key} ({zone_full_name}): Width={width:.3f}cm, Color={color_str}")

analyzer.visualize_analysis(results)


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from skimage import filters, segmentation
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
from typing import Tuple, Dict, List
import pywt

PAPER_DIAMETER_CM = 12.5  # Default diameter of chromatography paper in cm

class CircularChromatographyAnalyzer:
    """
    Circular Chromatography Analyzer based on the paper:
    "Chromatogram Image Pre-Processing and Feature Extraction for Automatic Soil Analysis"
    """
    
    def __init__(self, paper_diameter_cm: float = PAPER_DIAMETER_CM):
        """
        Initialize the analyzer.
        
        Args:
            paper_diameter_cm: Physical diameter of the chromatography paper in centimeters
        """
        self.paper_diameter_cm = paper_diameter_cm
        self.resolution = None  # Will be calculated automatically
        self.center = None
        self.radius = None
        self.paper_radius_pixels = None
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Basic image preprocessing - noise reduction and enhancement.
        
        Args:
            image: Input chromatogram image
            
        Returns:
            Preprocessed image
        """
        if len(image.shape) == 3:
            # Convert to grayscale for some operations, but keep color for final analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(blurred)
        
        return enhanced
    
    def calculate_resolution(self, image: np.ndarray) -> float:
        """
        Calculate resolution based on image width equals paper diameter.
        
        Args:
            image: Input image
            
        Returns:
            Resolution in pixels per cm
        """
        # Image width corresponds to paper diameter
        image_width = image.shape[1]  # Width in pixels
        self.resolution = image_width / self.paper_diameter_cm
        
        # Store paper radius in pixels for visualization
        self.paper_radius_pixels = image_width / 2.0
        
        return self.resolution
    
    def detect_center(self, image: np.ndarray, threshold: int = 90, 
                     window_size: int = 200) -> Tuple[int, int]:
        """
        Detect the center of chromatogram using the hole detection method from the paper.
        
        Args:
            image: Input grayscale image
            threshold: Threshold value for binary conversion (paper uses 90)
            window_size: Size of the window to crop around center region
            
        Returns:
            Tuple of (x_center, y_center)
        """
        h, w = image.shape[:2]
        
        # Crop center region
        center_x, center_y = w // 2, h // 2
        half_window = window_size // 2
        
        # Ensure we don't go out of bounds
        x1 = max(0, center_x - half_window)
        y1 = max(0, center_y - half_window)
        x2 = min(w, center_x + half_window)
        y2 = min(h, center_y + half_window)
        
        cropped = image[y1:y2, x1:x2]
        
        # Binary thresholding as per equation (1) in the paper
        # B(x,y) = 0 if I(x,y) > α, = 1 if I(x,y) ≤ α
        binary = np.where(cropped <= threshold, 1, 0).astype(np.uint8)
        
        # Find center using projection method as per equation (2)
        # Sum along each row and column to find maximum projections
        row_sums = np.sum(binary, axis=1)  # Sum along columns (horizontal projection)
        col_sums = np.sum(binary, axis=0)  # Sum along rows (vertical projection)
        
        # Find coordinates with maximum projection
        yc_local = np.argmax(row_sums)
        xc_local = np.argmax(col_sums)
        
        # Convert back to full image coordinates
        xc = x1 + xc_local
        yc = y1 + yc_local
        
        self.center = (xc, yc)
        return xc, yc
    
    def transform_to_polar(self, image: np.ndarray, center: Tuple[int, int], 
                          max_radius: int = None) -> np.ndarray:
        """
        Transform image from Cartesian to polar coordinates as per equation (3).
        
        Args:
            image: Input image
            center: Center coordinates (xc, yc)
            max_radius: Maximum radius for transformation
            
        Returns:
            Polar transformed image
        """
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(h, w, 1)
            
        xc, yc = center
        
        # Set maximum radius if not provided
        if max_radius is None:
            max_radius = min(xc, yc, w-xc, h-yc)
        
        self.radius = max_radius
        
        # Create polar coordinate grid
        theta_steps = int(2 * np.pi * max_radius)  # Angular resolution
        r_steps = max_radius
        
        polar_image = np.zeros((r_steps, theta_steps, c), dtype=image.dtype)
        
        # Transform using equation (3): x(r,θ) = r*cos(θ) + xc, y(r,θ) = r*sin(θ) + yc
        for r in range(r_steps):
            for t in range(theta_steps):
                theta = 2 * np.pi * t / theta_steps
                
                # Convert polar to cartesian
                x = int(r * np.cos(theta) + xc)
                y = int(r * np.sin(theta) + yc)
                
                # Check bounds
                if 0 <= x < w and 0 <= y < h:
                    polar_image[r, t] = image[y, x]
        
        return polar_image.squeeze() if c == 1 else polar_image
    
    def dwt_features(self, image: np.ndarray, levels: int = 2) -> np.ndarray:
        """
        Extract DWT features as described in Section 3.1 of the paper.
        
        Args:
            image: Input image
            levels: Number of DWT decomposition levels
            
        Returns:
            Feature images
        """
        if len(image.shape) == 3:
            # Process each color channel separately
            features = []
            for c in range(image.shape[2]):
                channel_features = self._dwt_single_channel(image[:,:,c], levels)
                features.append(channel_features)
            return np.concatenate(features, axis=2)
        else:
            return self._dwt_single_channel(image, levels)
    
    def _dwt_single_channel(self, image: np.ndarray, levels: int = 2) -> np.ndarray:
        """
        Process single channel with DWT.
        """
        coeffs = pywt.wavedec2(image, 'db4', level=levels)
        
        # Reconstruct approximation and detail coefficients
        features = []
        
        # Process approximation coefficients (LL)
        approx = coeffs[0]
        features.append(self._local_energy(approx))
        
        # Process detail coefficients for each level
        for level_coeffs in coeffs[1:]:
            for detail in level_coeffs:  # LH, HL, HH
                features.append(self._local_energy(detail))
        
        # Resize all features to same size and stack
        target_size = features[0].shape
        resized_features = []
        
        for feat in features:
            if feat.shape != target_size:
                feat_resized = cv2.resize(feat, (target_size[1], target_size[0]))
            else:
                feat_resized = feat
            resized_features.append(feat_resized)
        
        return np.stack(resized_features, axis=2)
    
    def _local_energy(self, coeffs: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Compute local energy as per the paper: rectification + Gaussian smoothing.
        """
        # Rectification: take absolute value
        rectified = np.abs(coeffs)
        
        # Gaussian smoothing
        smoothed = ndimage.gaussian_filter(rectified, sigma=sigma)
        
        return smoothed
    
    def segment_regions(self, features: np.ndarray, n_clusters: int = 3, method: str = 'radial_guided') -> np.ndarray:
        """
        Segment regions using various clustering methods suitable for chromatography.
        
        Args:
            features: Feature images from DWT
            n_clusters: Number of clusters/regions
            method: Segmentation method ('radial_guided', 'radial_kmeans', 'gmm', 'fcm_approx')
            
        Returns:
            Segmentation mask
        """
        if method == 'radial_guided':
            return self._radial_guided_segmentation(features, n_clusters)
        elif method == 'radial_kmeans':
            return self._radial_kmeans_segmentation(features, n_clusters)
        elif method == 'gmm':
            return self._gmm_segmentation(features, n_clusters)
        elif method == 'fcm_approx':
            return self._fcm_approximation_segmentation(features, n_clusters)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def _radial_guided_segmentation(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Radial-guided segmentation that enforces concentric ring structure.
        """
        print(f"  🎯 Using radial-guided segmentation...")
        h, w = features.shape[:2]
        print(f"  📐 Feature map size: {h}x{w}")
        
        # Create radial coordinate map
        r_coords = np.arange(h)  # In polar coordinates, rows are radial distances
        
        # Combine DWT features with radial position information
        if len(features.shape) == 3:
            feature_vectors = features.reshape(-1, features.shape[2])
            print(f"  🔧 Using {features.shape[2]} DWT feature channels")
        else:
            feature_vectors = features.reshape(-1, 1)
            print(f"  🔧 Using single feature channel")
        
        # Add radial position as a strong feature
        radial_feature = np.repeat(r_coords, w).reshape(-1, 1)
        # Normalize and weight radial feature
        radial_feature_norm = radial_feature / np.max(radial_feature)
        radial_weight = 2.0  # Higher weight for radial structure
        
        # Combine features
        combined_features = np.hstack([
            feature_vectors,
            radial_feature_norm * radial_weight
        ])
        print(f"  🔗 Combined feature vector size: {combined_features.shape}")
        
        # Apply K-means clustering
        print(f"  🤖 Running K-means clustering for {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined_features)
        
        # Reshape and post-process
        segmentation = labels.reshape(h, w)
        
        # Sort regions by radial position (inner to outer)
        print(f"  📊 Sorting regions by radial position...")
        segmentation = self._sort_regions_by_radius(segmentation)
        
        # Apply morphological operations to clean up
        print(f"  🧹 Post-processing segmentation...")
        segmentation = self._postprocess_segmentation(segmentation)
        
        final_regions = len(np.unique(segmentation))
        print(f"  ✅ Segmentation complete - {final_regions} final regions")
        
        return segmentation
    
    def _radial_kmeans_segmentation(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Pure radial-based segmentation using K-means on radius values.
        """
        h, w = features.shape[:2]
        
        # Create radial distance map
        r_coords = np.arange(h).reshape(-1, 1)
        radial_features = np.repeat(r_coords, w, axis=1).reshape(-1, 1)
        
        # Apply K-means clustering on radial distances
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(radial_features)
        
        segmentation = labels.reshape(h, w)
        
        return segmentation
    
    def _gmm_segmentation(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Original Gaussian Mixture Model segmentation.
        """
        h, w = features.shape[:2]
        if len(features.shape) == 3:
            feature_vectors = features.reshape(-1, features.shape[2])
        else:
            feature_vectors = features.reshape(-1, 1)
        
        # Use Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(feature_vectors)
        
        segmentation = labels.reshape(h, w)
        return segmentation
    
    def _fcm_approximation_segmentation(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Fuzzy C-Means approximation using GMM with spatial regularization.
        """
        h, w = features.shape[:2]
        
        # Prepare feature vectors
        if len(features.shape) == 3:
            feature_vectors = features.reshape(-1, features.shape[2])
        else:
            feature_vectors = features.reshape(-1, 1)
        
        # Add spatial coordinates as features
        y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
        spatial_features = np.column_stack([
            x_coords.ravel() / h,  # Normalized radial position
            y_coords.ravel() / w   # Normalized angular position
        ])
        
        # Combine texture and spatial features
        spatial_weight = 0.5
        combined_features = np.hstack([
            feature_vectors,
            spatial_features * spatial_weight
        ])
        
        # Apply GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        labels = gmm.fit_predict(combined_features)
        
        segmentation = labels.reshape(h, w)
        segmentation = self._sort_regions_by_radius(segmentation)
        
        return segmentation
    
    def _sort_regions_by_radius(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Sort region labels by their average radial position (inner to outer).
        """
        h, w = segmentation.shape
        unique_labels = np.unique(segmentation)
        
        # Calculate average radius for each region
        region_radii = {}
        for label in unique_labels:
            mask = segmentation == label
            r_coords, _ = np.where(mask)
            if len(r_coords) > 0:
                region_radii[label] = np.mean(r_coords)
            else:
                region_radii[label] = h  # Put empty regions at the end
        
        # Sort regions by average radius
        sorted_labels = sorted(region_radii.keys(), key=lambda x: region_radii[x])
        
        # Create new segmentation with sorted labels
        new_segmentation = np.zeros_like(segmentation)
        for new_label, old_label in enumerate(sorted_labels):
            new_segmentation[segmentation == old_label] = new_label
        
        return new_segmentation
    
    def _postprocess_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Post-process segmentation to clean up noise and enforce structure.
        """
        # Apply median filter to reduce noise
        cleaned = ndimage.median_filter(segmentation, size=3)
        
        # Fill small holes in each region
        unique_labels = np.unique(cleaned)
        for label in unique_labels:
            mask = cleaned == label
            # Fill holes
            filled_mask = ndimage.binary_fill_holes(mask)
            cleaned[filled_mask] = label
        
        return cleaned
    
    def extract_features(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Extract features from segmented regions as per Section 3.3.
        
        Args:
            polar_image: Polar transformed image
            segmentation: Segmentation mask
            
        Returns:
            Dictionary of features for each region
        """
        print(f"  📊 Extracting features from segmented regions...")
        
        # Ensure segmentation has same size as polar image
        polar_h, polar_w = polar_image.shape[:2]
        seg_h, seg_w = segmentation.shape
        
        if (seg_h, seg_w) != (polar_h, polar_w):
            print(f"  🔧 Resizing segmentation from {seg_h}x{seg_w} to {polar_h}x{polar_w}")
            # Resize segmentation to match polar image dimensions
            segmentation_resized = cv2.resize(segmentation, (polar_w, polar_h), interpolation=cv2.INTER_NEAREST)
        else:
            segmentation_resized = segmentation
        
        features = {}
        unique_labels = np.unique(segmentation_resized)
        print(f"  🏷️  Processing {len(unique_labels)} regions...")
        
        for i, label in enumerate(unique_labels):
            print(f"    Region {i+1}/{len(unique_labels)} (label {label})...", end=" ")
            mask = segmentation_resized == label
            region_features = self._compute_region_features(polar_image, mask)
            features[f'region_{label}'] = region_features
            area = region_features['area_cm2']
            radius = region_features['mean_radius_cm']
            print(f"Area: {area:.2f}cm², Radius: {radius:.2f}cm")
            
        return features
    
    def _compute_region_features(self, polar_image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Compute essential features for a single region.
        """
        # Basic geometric features
        area_pixels = np.sum(mask)
        n_columns = polar_image.shape[1]
        width_pixels = area_pixels / n_columns if n_columns > 0 else 0
        
        # Convert to physical units (cm)
        area_cm2 = area_pixels / (self.resolution ** 2)
        width_cm = width_pixels / self.resolution
        
        # Radial position features
        r_coords, theta_coords = np.where(mask)
        if len(r_coords) > 0:
            mean_radius_pixels = np.mean(r_coords)
            mean_radius_cm = mean_radius_pixels / self.resolution
            radial_thickness_cm = (np.max(r_coords) - np.min(r_coords)) / self.resolution
            
            # Angular coverage
            unique_angles = len(np.unique(theta_coords))
            angular_coverage_degrees = (unique_angles / n_columns) * 360 if n_columns > 0 else 0
        else:
            mean_radius_cm = radial_thickness_cm = angular_coverage_degrees = 0
        
        # Color features (simplified)
        if len(polar_image.shape) == 3:
            color_features = {}
            for c, channel in enumerate(['red', 'green', 'blue']):  # polar_image is RGB (converted from BGR)
                channel_values = polar_image[:,:,c][mask]
                color_features[channel] = np.mean(channel_values) if len(channel_values) > 0 else 0
        else:
            gray_values = polar_image[mask] if area_pixels > 0 else np.array([0])
            color_features = {'gray': np.mean(gray_values)}
        
        # Intensity statistics
        if area_pixels > 0:
            if len(polar_image.shape) == 3:
                # Extract RGB values for masked pixels and average across channels
                masked_pixels = polar_image[mask]  # Shape: (num_pixels, 3)
                gray_values = np.mean(masked_pixels, axis=1)  # Average across RGB channels
            else:
                gray_values = polar_image[mask]
            
            mean_intensity = np.mean(gray_values)
            contrast = np.max(gray_values) - np.min(gray_values) if len(gray_values) > 1 else 0
        else:
            mean_intensity = contrast = 0
        
        return {
            'area_cm2': area_cm2,
            'width_cm': width_cm,
            'mean_radius_cm': mean_radius_cm,
            'thickness_cm': radial_thickness_cm,
            'angular_coverage': angular_coverage_degrees,
            'mean_intensity': mean_intensity,
            'contrast': contrast,
            'color': color_features
        }
    
    def _estimate_perimeter(self, mask: np.ndarray) -> float:
        """
        Estimate the perimeter of a region using edge detection.
        """
        # Apply morphological gradient to find edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        return np.sum(edges > 0)
    
    def _calculate_eccentricity(self, mask: np.ndarray) -> float:
        """
        Calculate eccentricity using image moments.
        """
        try:
            # Calculate moments
            moments = cv2.moments(mask.astype(np.uint8))
            
            if moments['m00'] == 0:
                return 0
            
            # Central moments
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            
            # Calculate eigenvalues
            term1 = (mu20 + mu02) / 2
            term2 = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2) / 2
            
            lambda1 = term1 + term2
            lambda2 = term1 - term2
            
            if lambda1 <= 0:
                return 0
            
            eccentricity = np.sqrt(1 - (lambda2 / lambda1)) if lambda2 > 0 else 1
            return min(eccentricity, 1.0)  # Clamp to [0,1]
            
        except:
            return 0
    
    def _calculate_texture_features(self, polar_image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Calculate basic texture features for the region.
        """
        try:
            # Extract region pixels
            if len(polar_image.shape) == 3:
                region_pixels = cv2.cvtColor(polar_image, cv2.COLOR_RGB2GRAY)[mask]
            else:
                region_pixels = polar_image[mask]
            
            if len(region_pixels) == 0:
                return {'homogeneity': 0, 'energy': 0, 'entropy': 0}
            
            # Calculate histogram
            hist, _ = np.histogram(region_pixels, bins=16, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            
            # Remove zero probabilities to avoid log(0)
            hist = hist[hist > 0]
            
            # Calculate texture features
            homogeneity = np.sum(hist**2)  # Uniformity/Energy
            entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
            
            # Simple contrast measure
            if len(region_pixels) > 1:
                contrast = np.std(region_pixels) / (np.mean(region_pixels) + 1e-6)
            else:
                contrast = 0
            
            return {
                'homogeneity': float(homogeneity),
                'energy': float(homogeneity),  # Same as homogeneity
                'entropy': float(entropy),
                'contrast': float(contrast)
            }
            
        except:
            return {'homogeneity': 0, 'energy': 0, 'entropy': 0, 'contrast': 0}
    
    def analyze_radial_features(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Analyze radial features including channel development and spike development.
        
        Args:
            polar_image: Polar transformed image
            segmentation: Segmentation mask
            
        Returns:
            Dictionary containing radial feature analysis
        """
        print(f"  🔍 Analyzing radial features (channels & spikes)...")
        
        radial_features = {}
        
        # Analyze channel development
        channel_analysis = self._analyze_channel_development(polar_image, segmentation)
        radial_features['channel_development'] = channel_analysis
        
        # Analyze spike development
        spike_analysis = self._analyze_spike_development(polar_image, segmentation)
        radial_features['spike_development'] = spike_analysis
        
        # Analyze radial uniformity
        uniformity_analysis = self._analyze_radial_uniformity(polar_image, segmentation)
        radial_features['radial_uniformity'] = uniformity_analysis
        
        # Analyze directional preferences
        directional_analysis = self._analyze_directional_patterns(polar_image, segmentation)
        radial_features['directional_patterns'] = directional_analysis
        
        print(f"  ✅ Radial feature analysis complete")
        return radial_features
    
    def _analyze_channel_development(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Analyze channel development - continuous radial paths from center outward.
        Channels represent continuous migration paths of substances.
        """
        print(f"    📡 Analyzing channel development...")
        
        h, w = polar_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(polar_image.shape) == 3:
            gray_polar = cv2.cvtColor(polar_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_polar = polar_image.copy()
        
        # Apply edge detection to find boundaries
        edges = cv2.Canny(gray_polar.astype(np.uint8), 50, 150)
        
        # Analyze radial continuity for each angular position
        channel_metrics = {}
        angular_positions = np.arange(0, w, max(1, w // 360))  # Sample every degree
        
        channel_lengths = []
        channel_intensities = []
        channel_continuities = []
        
        for theta_idx in angular_positions:
            if theta_idx >= w:
                continue
                
            # Extract radial profile at this angle
            radial_profile = gray_polar[:, theta_idx]
            edge_profile = edges[:, theta_idx]
            
            # Find continuous segments (channels)
            channels = self._find_radial_channels(radial_profile, edge_profile)
            
            for channel in channels:
                start_r, end_r, intensity, continuity = channel
                channel_length = (end_r - start_r) / self.resolution  # Convert to cm
                
                channel_lengths.append(channel_length)
                channel_intensities.append(intensity)
                channel_continuities.append(continuity)
        
        # Calculate channel development metrics
        if channel_lengths:
            channel_metrics = {
                'total_channels': len(channel_lengths),
                'avg_channel_length_cm': np.mean(channel_lengths),
                'max_channel_length_cm': np.max(channel_lengths),
                'avg_channel_intensity': np.mean(channel_intensities),
                'avg_continuity': np.mean(channel_continuities),
                'channel_density': len(channel_lengths) / len(angular_positions),
                'channel_length_std': np.std(channel_lengths)
            }
        else:
            channel_metrics = {
                'total_channels': 0,
                'avg_channel_length_cm': 0,
                'max_channel_length_cm': 0,
                'avg_channel_intensity': 0,
                'avg_continuity': 0,
                'channel_density': 0,
                'channel_length_std': 0
            }
        
        print(f"      Channels found: {channel_metrics['total_channels']}")
        print(f"      Avg length: {channel_metrics['avg_channel_length_cm']:.2f}cm")
        
        return channel_metrics
    
    def _find_radial_channels(self, radial_profile: np.ndarray, edge_profile: np.ndarray, 
                             min_length: int = 5, intensity_threshold: float = 0.1) -> List[Tuple]:
        """
        Find continuous radial channels in a single angular direction.
        
        Args:
            radial_profile: Intensity values along radius
            edge_profile: Edge detection results along radius
            min_length: Minimum length for a valid channel
            intensity_threshold: Minimum intensity variation for channel detection
        
        Returns:
            List of tuples (start_r, end_r, avg_intensity, continuity_score)
        """
        channels = []
        
        # Smooth the profile to reduce noise
        smoothed = ndimage.gaussian_filter1d(radial_profile.astype(np.float32), sigma=1.0)
        
        # Find regions with significant intensity variation
        intensity_grad = np.abs(np.gradient(smoothed))
        active_regions = intensity_grad > (np.mean(intensity_grad) + intensity_threshold * np.std(intensity_grad))
        
        # Find continuous segments
        segments = []
        start = None
        
        for i, is_active in enumerate(active_regions):
            if is_active and start is None:
                start = i
            elif not is_active and start is not None:
                if i - start >= min_length:
                    segments.append((start, i))
                start = None
        
        # Handle case where segment extends to end
        if start is not None and len(active_regions) - start >= min_length:
            segments.append((start, len(active_regions)))
        
        # Calculate metrics for each segment
        for start_r, end_r in segments:
            segment_intensities = smoothed[start_r:end_r]
            avg_intensity = np.mean(segment_intensities)
            
            # Calculate continuity score (lower edge density = more continuous)
            segment_edges = edge_profile[start_r:end_r]
            continuity = 1.0 - (np.sum(segment_edges) / len(segment_edges)) if len(segment_edges) > 0 else 0
            
            channels.append((start_r, end_r, avg_intensity, continuity))
        
        return channels
    
    def _analyze_spike_development(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Analyze spike development - sharp, localized radial features.
        Spikes represent isolated migration points or contamination.
        """
        print(f"    🔺 Analyzing spike development...")
        
        h, w = polar_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(polar_image.shape) == 3:
            gray_polar = cv2.cvtColor(polar_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_polar = polar_image.copy()
        
        # Use Laplacian of Gaussian for spike detection
        # First apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_polar.astype(np.float32), (5, 5), 1.0)
        
        # Then apply Laplacian
        log_response = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        
        # Find local maxima (spikes)
        local_maxima = (log_response == ndimage.maximum_filter(log_response, size=5))
        
        # Threshold to get significant spikes
        threshold = np.mean(np.abs(log_response)) + 2 * np.std(np.abs(log_response))
        spikes = local_maxima & (np.abs(log_response) > threshold)
        
        # Analyze spike characteristics
        spike_locations = np.where(spikes)
        spike_radii = spike_locations[0] / self.resolution  # Convert to cm
        spike_angles = spike_locations[1] * 360.0 / w  # Convert to degrees
        spike_intensities = log_response[spike_locations]
        
        # Group spikes by radial regions
        spike_metrics = {
            'total_spikes': len(spike_radii),
            'spike_density': len(spike_radii) / (h * w) * 10000,  # Spikes per 10k pixels
            'avg_spike_intensity': np.mean(np.abs(spike_intensities)) if len(spike_intensities) > 0 else 0,
            'spike_radial_distribution': self._analyze_spike_distribution(spike_radii, spike_angles),
            'spike_clustering': self._analyze_spike_clustering(spike_radii, spike_angles)
        }
        
        print(f"      Spikes found: {spike_metrics['total_spikes']}")
        print(f"      Density: {spike_metrics['spike_density']:.2f} spikes/10k pixels")
        
        return spike_metrics
    
    def _analyze_spike_distribution(self, spike_radii: np.ndarray, spike_angles: np.ndarray) -> Dict:
        """
        Analyze the spatial distribution of spikes.
        """
        if len(spike_radii) == 0:
            return {'radial_uniformity': 1.0, 'angular_uniformity': 1.0, 'preferred_radius': 0}
        
        # Radial distribution analysis
        radial_bins = np.linspace(0, np.max(spike_radii) if len(spike_radii) > 0 else 1, 10)
        radial_hist, _ = np.histogram(spike_radii, bins=radial_bins)
        radial_uniformity = 1.0 - (np.std(radial_hist) / (np.mean(radial_hist) + 1e-6))
        
        # Angular distribution analysis
        angular_bins = np.linspace(0, 360, 36)  # 10-degree bins
        angular_hist, _ = np.histogram(spike_angles, bins=angular_bins)
        angular_uniformity = 1.0 - (np.std(angular_hist) / (np.mean(angular_hist) + 1e-6))
        
        # Find preferred radius (where most spikes occur)
        if len(spike_radii) > 0:
            preferred_radius_idx = np.argmax(radial_hist)
            preferred_radius = (radial_bins[preferred_radius_idx] + radial_bins[preferred_radius_idx + 1]) / 2
        else:
            preferred_radius = 0
        
        return {
            'radial_uniformity': max(0, min(1, radial_uniformity)),
            'angular_uniformity': max(0, min(1, angular_uniformity)),
            'preferred_radius': preferred_radius
        }
    
    def _analyze_spike_clustering(self, spike_radii: np.ndarray, spike_angles: np.ndarray) -> Dict:
        """
        Analyze clustering patterns in spike locations.
        """
        if len(spike_radii) < 2:
            return {'clustering_coefficient': 0, 'avg_nearest_neighbor_distance': 0}
        
        # Convert to Cartesian coordinates for distance calculation
        spike_x = spike_radii * np.cos(np.radians(spike_angles))
        spike_y = spike_radii * np.sin(np.radians(spike_angles))
        spike_coords = np.column_stack([spike_x, spike_y])
        
        # Calculate pairwise distances
        distances = squareform(pdist(spike_coords))
        
        # Remove self-distances (diagonal)
        np.fill_diagonal(distances, np.inf)
        
        # Find nearest neighbor distances
        nearest_distances = np.min(distances, axis=1)
        avg_nearest_distance = np.mean(nearest_distances)
        
        # Calculate clustering coefficient (simplified version)
        # Higher values indicate more clustering
        expected_distance = np.sqrt((np.max(spike_radii) ** 2) * np.pi / len(spike_radii))
        clustering_coefficient = max(0, 1 - (avg_nearest_distance / expected_distance))
        
        return {
            'clustering_coefficient': clustering_coefficient,
            'avg_nearest_neighbor_distance': avg_nearest_distance
        }
    
    def _analyze_radial_uniformity(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Analyze uniformity of development in radial direction.
        """
        print(f"    🎯 Analyzing radial uniformity...")
        
        h, w = polar_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(polar_image.shape) == 3:
            gray_polar = cv2.cvtColor(polar_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_polar = polar_image.copy()
        
        # Calculate radial profiles at different angles
        num_angles = min(72, w)  # Sample every 5 degrees or available resolution
        angle_indices = np.linspace(0, w-1, num_angles, dtype=int)
        
        radial_profiles = []
        for angle_idx in angle_indices:
            profile = gray_polar[:, angle_idx]
            radial_profiles.append(profile)
        
        radial_profiles = np.array(radial_profiles)
        
        # Calculate uniformity metrics
        # 1. Radial consistency: how similar are profiles across angles
        profile_correlations = []
        for i in range(len(radial_profiles)):
            for j in range(i+1, len(radial_profiles)):
                corr = np.corrcoef(radial_profiles[i], radial_profiles[j])[0, 1]
                if not np.isnan(corr):
                    profile_correlations.append(abs(corr))
        
        avg_correlation = np.mean(profile_correlations) if profile_correlations else 0
        
        # 2. Radial smoothness: how smooth is the average profile
        avg_profile = np.mean(radial_profiles, axis=0)
        profile_gradient = np.gradient(avg_profile)
        smoothness = 1.0 / (1.0 + np.std(profile_gradient))
        
        # 3. Development extent: how far does significant development extend
        profile_variance = np.var(radial_profiles, axis=0)
        significant_variance = profile_variance > (np.mean(profile_variance) * 0.5)
        development_extent = np.sum(significant_variance) / self.resolution  # Convert to cm
        
        uniformity_metrics = {
            'radial_consistency': avg_correlation,
            'radial_smoothness': smoothness,
            'development_extent_cm': development_extent,
            'profile_variance_mean': np.mean(profile_variance),
            'profile_variance_std': np.std(profile_variance)
        }
        
        print(f"      Consistency: {uniformity_metrics['radial_consistency']:.3f}")
        print(f"      Development extent: {uniformity_metrics['development_extent_cm']:.2f}cm")
        
        return uniformity_metrics
    
    def _analyze_directional_patterns(self, polar_image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Analyze directional patterns and asymmetries in development.
        """
        print(f"    🧭 Analyzing directional patterns...")
        
        h, w = polar_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(polar_image.shape) == 3:
            gray_polar = cv2.cvtColor(polar_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_polar = polar_image.copy()
        
        # Divide into angular sectors for analysis
        num_sectors = 8  # Analyze 8 directional sectors (45 degrees each)
        sector_size = w // num_sectors
        
        sector_metrics = []
        for i in range(num_sectors):
            start_angle = i * sector_size
            end_angle = min((i + 1) * sector_size, w)
            
            sector_data = gray_polar[:, start_angle:end_angle]
            sector_mean = np.mean(sector_data)
            sector_std = np.std(sector_data)
            sector_development = np.sum(sector_data > np.mean(gray_polar))
            
            sector_metrics.append({
                'angle_range': (start_angle * 360 / w, end_angle * 360 / w),
                'mean_intensity': sector_mean,
                'intensity_std': sector_std,
                'development_pixels': sector_development
            })
        
        # Calculate directional asymmetry
        intensities = [s['mean_intensity'] for s in sector_metrics]
        developments = [s['development_pixels'] for s in sector_metrics]
        
        intensity_asymmetry = np.std(intensities) / (np.mean(intensities) + 1e-6)
        development_asymmetry = np.std(developments) / (np.mean(developments) + 1e-6)
        
        # Find dominant direction
        max_development_idx = np.argmax(developments)
        dominant_direction = (sector_metrics[max_development_idx]['angle_range'][0] + 
                            sector_metrics[max_development_idx]['angle_range'][1]) / 2
        
        # Calculate symmetry score (compare opposite sectors)
        symmetry_scores = []
        for i in range(num_sectors // 2):
            opposite_i = (i + num_sectors // 2) % num_sectors
            symmetry = 1.0 - abs(intensities[i] - intensities[opposite_i]) / (max(intensities[i], intensities[opposite_i]) + 1e-6)
            symmetry_scores.append(symmetry)
        
        directional_metrics = {
            'intensity_asymmetry': intensity_asymmetry,
            'development_asymmetry': development_asymmetry,
            'dominant_direction_degrees': dominant_direction,
            'symmetry_score': np.mean(symmetry_scores),
            'sector_analysis': sector_metrics
        }
        
        print(f"      Asymmetry: {directional_metrics['intensity_asymmetry']:.3f}")
        print(f"      Dominant direction: {directional_metrics['dominant_direction_degrees']:.1f}°")
        
        return directional_metrics
    
    def analyze_radial_features_standalone(self, image_path: str, n_regions: int = 5, 
                                         segmentation_method: str = 'fcm_approx') -> Dict:
        """
        Standalone radial feature analysis without full chromatogram processing.
        Useful for focused radial analysis of existing chromatograms.
        
        Args:
            image_path: Path to chromatogram image
            n_regions: Number of regions for segmentation
            segmentation_method: Segmentation method to use
            
        Returns:
            Dictionary containing only radial feature analysis
        """
        print(f"🔍 Starting focused radial feature analysis...")
        
        # Run basic analysis without visualization
        results = self.analyze_chromatogram(image_path, n_regions, 
                                          segmentation_method=segmentation_method)
        
        # Extract and return only radial features
        radial_features = results.get('radial_features', {})
        
        # Show radial feature visualization
        print(f"📊 Generating radial feature visualization...")
        temp_results = {'radial_features': radial_features}
        self.visualize_radial_features(temp_results)
        
        return radial_features
    
    def analyze_chromatogram(self, image_path: str, n_regions: int = 5, 
                           segmentation_method: str = 'fcm_approx') -> Dict:
        """
        Complete analysis pipeline for a chromatogram image.
        
        Args:
            image_path: Path to chromatogram image
            n_regions: Expected number of regions (default: 5 for detailed analysis)
            segmentation_method: Method for segmentation ('radial_guided', 'radial_kmeans', 'gmm', 'fcm_approx')
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"🔬 Starting chromatogram analysis...")
        print(f"📁 Loading image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"✅ Image loaded successfully - Size: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate resolution based on image width = paper diameter
        print(f"📏 Calculating resolution...")
        resolution = self.calculate_resolution(image)
        print(f"✅ Resolution: {resolution:.1f} pixels/cm (Paper: {self.paper_diameter_cm}cm)")
        
        # Preprocess
        print(f"🔧 Preprocessing image (noise reduction & contrast enhancement)...")
        preprocessed = self.preprocess_image(image)
        
        # Detect center
        print(f"🎯 Detecting chromatogram center...")
        center_x, center_y = self.detect_center(preprocessed)
        print(f"✅ Center detected at: ({center_x}, {center_y})")
        
        # Transform to polar coordinates
        print(f"🔄 Transforming to polar coordinates...")
        polar_image = self.transform_to_polar(image_rgb, (center_x, center_y))
        print(f"✅ Polar transformation complete - Size: {polar_image.shape[0]}x{polar_image.shape[1]}")
        
        # Extract DWT features
        print(f"🌊 Extracting DWT features...")
        features = self.dwt_features(polar_image)
        print(f"✅ DWT features extracted - Shape: {features.shape}")
        
        # Segment regions using selected method
        print(f"🎨 Segmenting regions using '{segmentation_method}' method...")
        print(f"📊 Target regions: {n_regions}")
        segmentation = self.segment_regions(features, n_regions, method=segmentation_method)
        unique_regions = len(np.unique(segmentation))
        print(f"✅ Segmentation complete - Found {unique_regions} regions")
        
        # Extract quantitative features
        print(f"📈 Extracting quantitative features from regions...")
        region_features = self.extract_features(polar_image, segmentation)
        print(f"✅ Feature extraction complete - {len(region_features)} regions analyzed")
        
        # Sort regions by mean radius and assign zone names to inner 3 regions
        print(f"🏷️  Assigning zone names to inner 3 regions...")
        sorted_region_items = sorted(region_features.items(), key=lambda x: x[1]['mean_radius_cm'])
        
        # Create new features dictionary with zone names
        zone_features = {}
        zone_names = ["CZ", "MZ", "OZ"]  # Central, Median, Outer zones
        zone_full_names = ["Central zone", "Median zone", "Outer zone"]
        
        for i, (original_key, features) in enumerate(sorted_region_items):
            if i < 3:
                # Assign zone names to inner 3 regions
                zone_key = zone_names[i]
                features['zone_name'] = zone_names[i]
                features['zone_full_name'] = zone_full_names[i]
                features['zone_index'] = i
                zone_features[zone_key] = features
                print(f"    {original_key} → {zone_key} ({zone_full_names[i]})")
            else:
                # Keep original naming for outer regions
                features['zone_name'] = None
                features['zone_full_name'] = f"Region {i+1}"
                features['zone_index'] = i
                zone_features[original_key] = features
        
        print(f"✅ Zone naming complete - 3 primary zones identified")
        
        # Analyze radial features (channels, spikes, uniformity)
        print(f"🔍 Analyzing radial features (channels & spikes)...")
        radial_features = self.analyze_radial_features(polar_image, segmentation)
        print(f"✅ Radial feature analysis complete")
        
        # Prepare results with zone-named features
        print(f"📦 Preparing analysis results...")
        results = {
            'center': (center_x, center_y),
            'polar_image': polar_image,
            'segmentation': segmentation,
            'features': zone_features,  # Now contains zone-named features
            'radial_features': radial_features,
            'original_image': image_rgb,
            'zone_mapping': {zone_names[i]: zone_full_names[i] for i in range(3)}
        }
        
        print(f"🎉 Analysis complete!")
        return results
    
    def visualize_analysis(self, results: Dict, show_radial_features: bool = True):
        """
        Visualize the analysis results.
        
        Args:
            results: Dictionary containing analysis results from analyze_chromatogram()
            show_radial_features: Whether to show radial feature visualization
        """
        print(f"📊 Generating visualization...")
        self.visualize_results(results)
        
        if show_radial_features:
            self.visualize_radial_features(results)
    
    def visualize_results(self, results: Dict):
        """
        Visualize the analysis results with custom layout:
        - Upper left: Original image with regions
        - Upper right: Polar and segmentation plots stacked
        - Bottom left: Region analysis table
        - Bottom right: Empty space
        """
        # Create a custom layout: 2x2 grid with subdivisions
        fig = plt.figure(figsize=(16, 12))
        
        # Adjust subplot spacing for the new layout
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.06, right=0.95, 
                           hspace=0.25, wspace=0.2)
        
        # Define grid layout: 4 rows, 2 columns for proper subdivision
        # Upper left: Original image (spans 2 rows)
        # Upper right: Polar and segmentation (1 row each)
        # Bottom left: Features table (spans 2 rows)
        # Bottom right: Empty space (spans 2 rows)
        
        # Original image with region markers (upper left, spans 2 rows)
        ax_original = plt.subplot2grid((4, 2), (0, 0), rowspan=2)
        ax_original.imshow(results['original_image'])
        
        # Add region boundaries and labels on the original image
        self._overlay_regions_on_original(ax_original, results)
        
        if hasattr(self, 'paper_radius_pixels') and self.paper_radius_pixels:
            # Add resolution text
            ax_original.text(0.02, 0.98, f'Paper: {self.paper_diameter_cm}cm\nResolution: {self.resolution:.1f} px/cm', 
                         transform=ax_original.transAxes, fontsize=10, 
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                         verticalalignment='top')
        
        ax_original.set_title('Original Image with Region Markers')
        ax_original.axis('off')
        
        # Upper right: Polar plot (top of right column)
        ax_polar = plt.subplot2grid((4, 2), (0, 1))
        ax_polar.imshow(results['polar_image'])
        ax_polar.set_title('Polar Transformed Image')
        ax_polar.set_xlabel('θ (angle)')
        ax_polar.set_ylabel('r (radius)')
        
        # Upper right: Segmentation plot (second row of right column)
        ax_segmentation = plt.subplot2grid((4, 2), (1, 1))
        
        # Segmentation with consistent color coding
        # Create custom colormap that matches the overlay colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Sort segmentation to match the region ordering used in overlay
        sorted_regions = sorted(results['features'].items(), 
                              key=lambda x: x[1]['mean_radius_cm'])
        
        # Create a color-mapped segmentation image
        segmentation_colored = np.zeros((*results['segmentation'].shape, 3))
        unique_labels = np.unique(results['segmentation'])
        
        # Map each region to its corresponding color
        for i, (region_key, region_features) in enumerate(sorted_regions):
            if i < len(colors):
                # Extract region number from segmentation based on zone_index
                if 'zone_index' in region_features:
                    region_label = region_features['zone_index']
                else:
                    # Fallback: extract from region_X format
                    try:
                        region_label = int(region_key.split('_')[1])
                    except (IndexError, ValueError):
                        region_label = i  # Use index as fallback
                
                # Convert hex color to RGB
                hex_color = colors[i].lstrip('#')
                rgb_color = tuple(int(hex_color[j:j+2], 16)/255.0 for j in (0, 2, 4))
                
                # Apply color to segmentation
                mask = results['segmentation'] == region_label
                segmentation_colored[mask] = rgb_color
        
        ax_segmentation.imshow(segmentation_colored)
        ax_segmentation.set_title('Segmentation (Color-Coded)')
        ax_segmentation.set_xlabel('θ (angle)')
        ax_segmentation.set_ylabel('r (radius)')
        
        # Bottom left: Features table (spans 2 rows)
        ax_features = plt.subplot2grid((4, 2), (2, 0), rowspan=2)
        ax_features.axis('off')
        
        # Bottom right: Empty area (spans 2 rows)
        ax_claude = plt.subplot2grid((4, 2), (2, 1), rowspan=2)
        ax_claude.axis('off')
        
        # Feature summary table with simplified features for all regions
        feature_text = "Region Analysis (5 Regions):\n" + "="*40 + "\n\n"
        
        # Sort regions by mean radius for better display
        sorted_regions = sorted(results['features'].items(), 
                              key=lambda x: x[1]['mean_radius_cm'])
        
        # Create 2-column layout
        col1_text = ""
        col2_text = ""
        
        for i, (region, features) in enumerate(sorted_regions):
            region_num = i + 1
            # Use zone_full_name if available, otherwise use region key
            display_name = features.get('zone_full_name', region)
            region_text = f"Region {region_num} ({display_name}):\n"
            region_text += f"  Width: {features['width_cm']:.3f} cm\n"
            region_text += f"  Radius: {features['mean_radius_cm']:.2f} cm\n"
            region_text += f"  Thickness: {features['thickness_cm']:.2f} cm\n"
            region_text += f"  Intensity: {features['mean_intensity']:.0f}\n"
            
            # Color info (simplified) - we'll add rectangles separately
            if 'red' in features['color']:
                r = features['color']['red']
                g = features['color']['green'] 
                b = features['color']['blue']
                region_text += f"  RGB: ({r:.0f}, {g:.0f}, {b:.0f})\n"
            else:
                gray = features['color']['gray']
                region_text += f"  Gray: {gray:.0f}\n"
            
            region_text += "\n"
            
            # Alternate between columns
            if i < 3:  # First 3 regions in left column
                col1_text += region_text
            else:  # Last 2 regions in right column
                col2_text += region_text
        
        # Combine both columns with proper spacing
        combined_text = feature_text
        lines1 = col1_text.split('\n')
        lines2 = col2_text.split('\n')
        
        # Pad the shorter column with empty lines
        max_lines = max(len(lines1), len(lines2))
        lines1.extend([''] * (max_lines - len(lines1)))
        lines2.extend([''] * (max_lines - len(lines2)))
        
        # Combine lines side by side
        for line1, line2 in zip(lines1, lines2):
            combined_text += f"{line1:<25} {line2}\n"
        
        ax_features.text(0.05, 0.95, combined_text, transform=ax_features.transAxes, 
                      fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        # Add colored rectangles next to RGB values
        for i, (region, features) in enumerate(sorted_regions):
            region_num = i + 1
            
            # Calculate position for the color rectangle
            if i < 3:  # First 3 regions in left column
                x_pos = 0.35  # Position after RGB text in left column
                y_start = 0.85  # Starting Y position
                y_pos = y_start - (i * 0.14)  # Each region takes about 0.14 height
            else:  # Last 2 regions in right column
                x_pos = 0.85  # Position after RGB text in right column
                y_start = 0.85  # Starting Y position
                y_pos = y_start - ((i-3) * 0.14)  # Each region takes about 0.14 height
            
            # Get the color values
            if 'red' in features['color']:
                r = features['color']['red'] / 255.0  # Normalize to 0-1
                g = features['color']['green'] / 255.0
                b = features['color']['blue'] / 255.0
                color = (r, g, b)
            else:
                gray = features['color']['gray'] / 255.0
                color = (gray, gray, gray)
            
            # Add colored rectangle
            rect = plt.Rectangle((x_pos, y_pos - 0.015), 0.03, 0.02, 
                               facecolor=color, edgecolor='black', linewidth=0.5,
                               transform=ax_features.transAxes)
            ax_features.add_patch(rect)
        
        ax_features.set_title('Extracted Features')
        
        # Don't use tight_layout since we're using custom subplots_adjust
        plt.show()

    def _overlay_regions_on_original(self, ax, results):
        """
        Overlay simple circular region boundaries based on computed features.
        """
        if not hasattr(self, 'center') or self.center is None:
            return
            
        center_x, center_y = self.center
        
        # Sort regions by mean radius for consistent display
        sorted_regions = sorted(results['features'].items(), 
                              key=lambda x: x[1]['mean_radius_cm'])
        
        # Define colors for each region (matching tab10 colormap)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Draw simple circular boundaries at mean radius for each region
        for i, (region, features) in enumerate(sorted_regions):
            if i < len(colors):
                color = colors[i]
                region_num = i + 1
                
                # Use zone name if available
                if 'zone_name' in features and features['zone_name']:
                    display_label = features['zone_name']
                else:
                    display_label = f'R{region_num}'
                
                # Get the computed mean radius in cm, convert to pixels
                mean_radius_cm = features['mean_radius_cm']
                mean_radius_px = mean_radius_cm * self.resolution
                
                # Draw single circle at mean radius
                mean_circle = plt.Circle((center_x, center_y), mean_radius_px, 
                                       fill=False, color=color, linewidth=2, 
                                       linestyle='-', alpha=0.8)
                ax.add_patch(mean_circle)
                
                # Add region label
                label_x = center_x + mean_radius_px * 0.7  # 45 degree angle
                label_y = center_y - mean_radius_px * 0.7
                
                ax.text(label_x, label_y, display_label, 
                       fontsize=12, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                               edgecolor=color, alpha=0.8),
                       ha='center', va='center')
        
        # Mark the center point
        ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3)
        
        # Add legend
        legend_elements = []
        for i, (region, features) in enumerate(sorted_regions):
            if i < len(colors):
                # Use zone full name if available, otherwise generic region name
                if 'zone_full_name' in features:
                    label_text = features['zone_full_name']
                else:
                    label_text = f'Region {i+1}'
                
                legend_elements.append(plt.Line2D([0], [0], color=colors[i], lw=2, 
                                                label=label_text))
        
        legend_elements.append(plt.Line2D([0], [0], marker='+', color='red', 
                                        linestyle='None', markersize=10, 
                                        markeredgewidth=2, label='Center'))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                 bbox_to_anchor=(0.98, 0.98))

    def visualize_radial_features(self, results: Dict):
        """
        Visualize radial feature analysis results.
        """
        if 'radial_features' not in results:
            print("⚠️  No radial features found in results")
            return
            
        radial_features = results['radial_features']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Radial Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Channel Development Visualization
        if 'channel_development' in radial_features:
            ax = axes[0, 0]
            channel_data = radial_features['channel_development']
            
            # Create bar chart of channel metrics
            metrics = ['Total Channels', 'Avg Length (cm)', 'Max Length (cm)', 
                      'Channel Density', 'Avg Continuity']
            values = [
                channel_data['total_channels'],
                channel_data['avg_channel_length_cm'],
                channel_data['max_channel_length_cm'],
                channel_data['channel_density'] * 100,  # Scale for visibility
                channel_data['avg_continuity'] * 100    # Convert to percentage
            ]
            
            bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_title('Channel Development Metrics')
            ax.set_ylabel('Value')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Spike Development Visualization
        if 'spike_development' in radial_features:
            ax = axes[0, 1]
            spike_data = radial_features['spike_development']
            
            # Create donut chart for spike distribution
            if spike_data['total_spikes'] > 0:
                spike_dist = spike_data['spike_radial_distribution']
                
                # Prepare data for visualization
                labels = ['Total Spikes', 'Density (×10k)', 'Avg Intensity', 'Clustering']
                values = [
                    spike_data['total_spikes'],
                    spike_data['spike_density'],
                    spike_data['avg_spike_intensity'],
                    spike_data['spike_clustering']['clustering_coefficient'] * 100
                ]
                
                bars = ax.bar(labels, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
                ax.set_title('Spike Development Metrics')
                ax.set_ylabel('Value')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Spikes Detected', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title('Spike Development Metrics')
        
        # 3. Radial Uniformity Visualization
        if 'radial_uniformity' in radial_features:
            ax = axes[0, 2]
            uniformity_data = radial_features['radial_uniformity']
            
            # Radar chart for uniformity metrics
            metrics = ['Consistency', 'Smoothness', 'Development\nExtent (cm)']
            values = [
                uniformity_data['radial_consistency'] * 100,
                uniformity_data['radial_smoothness'] * 100,
                min(uniformity_data['development_extent_cm'], 10) * 10  # Scale and cap
            ]
            
            bars = ax.bar(metrics, values, color=['#ffd93d', '#6bcf7f', '#4d96ff'])
            ax.set_title('Radial Uniformity Metrics')
            ax.set_ylabel('Score (%)')
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Directional Patterns Visualization
        if 'directional_patterns' in radial_features:
            ax = axes[1, 0]
            directional_data = radial_features['directional_patterns']
            
            # Polar plot for directional analysis
            if 'sector_analysis' in directional_data:
                sector_analysis = directional_data['sector_analysis']
                
                # Convert to polar coordinates
                angles = [np.radians((s['angle_range'][0] + s['angle_range'][1]) / 2) 
                         for s in sector_analysis]
                intensities = [s['mean_intensity'] for s in sector_analysis]
                
                # Create polar plot
                ax = plt.subplot(2, 3, 4, projection='polar')
                ax.plot(angles, intensities, 'o-', linewidth=2, markersize=8, color='#e74c3c')
                ax.fill(angles, intensities, alpha=0.25, color='#e74c3c')
                ax.set_title('Directional Intensity Pattern', pad=20)
                ax.set_ylim(0, max(intensities) * 1.1 if intensities else 1)
                
                # Mark dominant direction
                dominant_angle = np.radians(directional_data['dominant_direction_degrees'])
                ax.annotate('Dominant', xy=(dominant_angle, max(intensities)), 
                           xytext=(dominant_angle, max(intensities) * 1.3),
                           ha='center', va='center', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # 5. Summary Statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = "🔍 RADIAL FEATURE SUMMARY\n" + "="*40 + "\n\n"
        
        if 'channel_development' in radial_features:
            ch = radial_features['channel_development']
            summary_text += f"📡 CHANNEL DEVELOPMENT:\n"
            summary_text += f"  • Total channels: {ch['total_channels']}\n"
            summary_text += f"  • Average length: {ch['avg_channel_length_cm']:.2f} cm\n"
            summary_text += f"  • Channel density: {ch['channel_density']:.3f}\n"
            summary_text += f"  • Continuity score: {ch['avg_continuity']:.3f}\n\n"
        
        if 'spike_development' in radial_features:
            sp = radial_features['spike_development']
            summary_text += f"🔺 SPIKE DEVELOPMENT:\n"
            summary_text += f"  • Total spikes: {sp['total_spikes']}\n"
            summary_text += f"  • Spike density: {sp['spike_density']:.2f}/10k px\n"
            summary_text += f"  • Average intensity: {sp['avg_spike_intensity']:.1f}\n"
            if sp['total_spikes'] > 0:
                summary_text += f"  • Clustering: {sp['spike_clustering']['clustering_coefficient']:.3f}\n"
            summary_text += "\n"
        
        if 'radial_uniformity' in radial_features:
            ru = radial_features['radial_uniformity']
            summary_text += f"🎯 RADIAL UNIFORMITY:\n"
            summary_text += f"  • Consistency: {ru['radial_consistency']:.3f}\n"
            summary_text += f"  • Smoothness: {ru['radial_smoothness']:.3f}\n"
            summary_text += f"  • Development extent: {ru['development_extent_cm']:.2f} cm\n\n"
        
        if 'directional_patterns' in radial_features:
            dp = radial_features['directional_patterns']
            summary_text += f"🧭 DIRECTIONAL PATTERNS:\n"
            summary_text += f"  • Intensity asymmetry: {dp['intensity_asymmetry']:.3f}\n"
            summary_text += f"  • Development asymmetry: {dp['development_asymmetry']:.3f}\n"
            summary_text += f"  • Dominant direction: {dp['dominant_direction_degrees']:.1f}°\n"
            summary_text += f"  • Symmetry score: {dp['symmetry_score']:.3f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # 6. Interpretation Guide
        ax = axes[1, 2]
        ax.axis('off')
        
        interpretation_text = "📚 INTERPRETATION GUIDE\n" + "="*30 + "\n\n"
        interpretation_text += "📡 CHANNELS:\n"
        interpretation_text += "  • High density: Good separation\n"
        interpretation_text += "  • Long channels: Mobile phase migration\n"
        interpretation_text += "  • High continuity: Uniform development\n\n"
        
        interpretation_text += "🔺 SPIKES:\n"
        interpretation_text += "  • Few spikes: Clean separation\n"
        interpretation_text += "  • Many spikes: Contamination/noise\n"
        interpretation_text += "  • Clustering: Localized issues\n\n"
        
        interpretation_text += "🎯 UNIFORMITY:\n"
        interpretation_text += "  • High consistency: Even development\n"
        interpretation_text += "  • High smoothness: Gradual transitions\n"
        interpretation_text += "  • Large extent: Good separation range\n\n"
        
        interpretation_text += "🧭 DIRECTIONALITY:\n"
        interpretation_text += "  • Low asymmetry: Symmetric development\n"
        interpretation_text += "  • High symmetry: Balanced conditions\n"
        interpretation_text += "  • Dominant direction: Preferred flow\n"
        
        ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()


# Example usage and testing functions
def demo_analysis():
    """
    Demonstration of how to use the analyzer.
    """
    print("🧪 CHROMATOGRAPHY ANALYZER DEMO")
    print("=" * 50)
    
    # Initialize analyzer with known paper diameter
    print("🔧 Initializing analyzer...")
    analyzer = CircularChromatographyAnalyzer(paper_diameter_cm=10.0)
    print(f"✅ Analyzer ready - Paper diameter: {analyzer.paper_diameter_cm}cm")
    
    # Create a synthetic chromatogram for testing
    print("\n🎨 Creating synthetic chromatogram for testing...")
    synthetic_image = create_synthetic_chromatogram()
    
    # Save synthetic image for testing
    cv2.imwrite('synthetic_chromatogram.jpg', synthetic_image)
    print("✅ Synthetic chromatogram saved as 'synthetic_chromatogram.jpg'")
    
    # Analyze the chromatogram with improved segmentation
    try:
        print(f"\n🔬 Testing different segmentation methods:")
        print("=" * 50)
        
        methods = ['radial_guided', 'radial_kmeans', 'gmm', 'fcm_approx']
        
        for i, method in enumerate(methods):
            print(f"\n📊 [{i+1}/{len(methods)}] Testing '{method}' segmentation...")
            print("-" * 30)
            
            results = analyzer.analyze_chromatogram('synthetic_chromatogram.jpg', 
                                                  n_regions=5, visualize=False, 
                                                  segmentation_method=method)
            
            print(f"\n📋 RESULTS for {method}:")
            print(f"   Center detected at: {results['center']}")
            
            # Sort regions by radius for clearer display
            sorted_regions = sorted(results['features'].items(), 
                                  key=lambda x: x[1]['mean_radius_cm'])
            
            print(f"   Region Summary:")
            for j, (region, features) in enumerate(sorted_regions):
                print(f"     Region {j+1}: Area={features['area_cm2']:.2f}cm², " +
                      f"Radius={features['mean_radius_cm']:.2f}cm, " +
                      f"Intensity={features['mean_intensity']:.0f}")
        
        # Show visualization for the best method (gmm)
        print(f"\n🎭 Generating final visualization with 'gmm' method...")
        print("=" * 50)
        results = analyzer.analyze_chromatogram('synthetic_chromatogram.jpg', 
                                              n_regions=5, visualize=True, 
                                              segmentation_method='gmm')
        
        # Display radial feature summary
        if 'radial_features' in results:
            print(f"\n📊 RADIAL FEATURE ANALYSIS SUMMARY:")
            print("=" * 50)
            radial_features = results['radial_features']
            
            if 'channel_development' in radial_features:
                ch = radial_features['channel_development']
                print(f"📡 Channel Development:")
                print(f"   • Channels detected: {ch['total_channels']}")
                print(f"   • Average length: {ch['avg_channel_length_cm']:.2f} cm")
                print(f"   • Channel density: {ch['channel_density']:.3f}")
                print(f"   • Continuity score: {ch['avg_continuity']:.3f}")
            
            if 'spike_development' in radial_features:
                sp = radial_features['spike_development']
                print(f"\n🔺 Spike Development:")
                print(f"   • Spikes detected: {sp['total_spikes']}")
                print(f"   • Spike density: {sp['spike_density']:.2f} per 10k pixels")
                print(f"   • Average intensity: {sp['avg_spike_intensity']:.1f}")
                
            if 'radial_uniformity' in radial_features:
                ru = radial_features['radial_uniformity']
                print(f"\n🎯 Radial Uniformity:")
                print(f"   • Consistency score: {ru['radial_consistency']:.3f}")
                print(f"   • Smoothness score: {ru['radial_smoothness']:.3f}")
                print(f"   • Development extent: {ru['development_extent_cm']:.2f} cm")
                
            if 'directional_patterns' in radial_features:
                dp = radial_features['directional_patterns']
                print(f"\n🧭 Directional Patterns:")
                print(f"   • Intensity asymmetry: {dp['intensity_asymmetry']:.3f}")
                print(f"   • Dominant direction: {dp['dominant_direction_degrees']:.1f}°")
                print(f"   • Symmetry score: {dp['symmetry_score']:.3f}")
        
        print(f"\n🎉 Demo completed successfully!")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()


def demo_radial_analysis():
    """
    Focused demonstration of radial feature analysis.
    """
    print("🔍 RADIAL FEATURE ANALYSIS DEMO")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CircularChromatographyAnalyzer(paper_diameter_cm=10.0)
    
    # Create synthetic chromatogram if needed
    try:
        radial_features = analyzer.analyze_radial_features_standalone(
            'synthetic_chromatogram.jpg', n_regions=5, 
            segmentation_method='gmm'
        )
        
        print(f"\n✅ Radial analysis completed!")
        print(f"📊 Features analyzed: {list(radial_features.keys())}")
        
    except Exception as e:
        print(f"❌ Error in radial analysis: {e}")
        # Create synthetic image and try again
        print("🎨 Creating synthetic chromatogram...")
        synthetic_image = create_synthetic_chromatogram()
        cv2.imwrite('synthetic_chromatogram.jpg', synthetic_image)
        
        radial_features = analyzer.analyze_radial_features_standalone(
            'synthetic_chromatogram.jpg', n_regions=5, 
            segmentation_method='gmm'
        )


def create_synthetic_chromatogram(size: int = 400) -> np.ndarray:
    """
    Create a synthetic chromatogram for testing purposes with 5 distinct regions.
    """
    print(f"  🎨 Creating {size}x{size} synthetic chromatogram...")
    
    # Create blank image
    image = np.ones((size, size, 3), dtype=np.uint8) * 240  # Light background
    
    # Create center hole
    center = (size // 2, size // 2)
    cv2.circle(image, center, 15, (50, 50, 50), -1)  # Dark center hole
    
    # Create 5 concentric regions with different colors/intensities
    print(f"  🎯 Adding 5 concentric regions...")
    # Region 1: Innermost (excluding center hole)
    cv2.circle(image, center, 40, (200, 150, 100), 8)
    # Region 2: Inner
    cv2.circle(image, center, 65, (180, 160, 120), 10)
    # Region 3: Middle
    cv2.circle(image, center, 90, (170, 140, 110), 12)
    # Region 4: Outer
    cv2.circle(image, center, 120, (160, 130, 100), 14)
    # Region 5: Outermost
    cv2.circle(image, center, 150, (150, 120, 90), 16)
    
    # Add some noise and texture
    print(f"  🌪️  Adding noise and texture...")
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    print(f"  ✅ Synthetic chromatogram created")
    return image


if __name__ == "__main__":
    print("🔬 CHROMATOGRAPHY ANALYZER WITH RADIAL FEATURES")
    print("=" * 60)
    print("Choose analysis type:")
    print("1. Full chromatogram analysis (includes radial features)")
    print("2. Focused radial feature analysis only")
    print("3. Both analyses")
    
    choice = input("\nEnter choice (1-3) or press Enter for full analysis: ").strip()
    
    if choice == "2":
        # Run only radial analysis
        demo_radial_analysis()
    elif choice == "3":
        # Run both
        demo_analysis()
        print("\n" + "="*60)
        demo_radial_analysis()
    else:
        # Run full analysis (default)
        demo_analysis()

"""
Logo Feature Extraction Module
Extracts pHash, RGB histograms, Hu moments, and ORB descriptors from logos
"""

import os
import cv2
import numpy as np
import imagehash
from PIL import Image
import pickle
from tqdm import tqdm
import cairosvg
from io import BytesIO

class LogoFeatureExtractor:
    def __init__(self, logo_dir, white_threshold=240):
        """
        Initialize feature extractor
        
        Args:
            logo_dir (str): Directory containing logo images
            white_threshold (int): Threshold for white background masking
        """
        self.logo_dir = logo_dir
        self.white_threshold = white_threshold
        self.features = {}
        self.orb_descriptors = {}
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        
    def _convert_svg_to_png(self, svg_path):
        """Convert SVG to PNG in memory"""
        try:
            png_data = cairosvg.svg2png(url=svg_path, output_width=256, output_height=256)
            return Image.open(BytesIO(png_data)).convert('RGB')
        except Exception as e:
            print(f"Error converting SVG {svg_path}: {e}")
            return None
    
    def _load_image(self, image_path):
        """Load and preprocess image"""
        try:
            if image_path.lower().endswith('.svg'):
                pil_image = self._convert_svg_to_png(image_path)
                if pil_image is None:
                    return None, None
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                pil_image = Image.open(image_path).convert('RGB')
                cv_image = cv2.imread(image_path)
                
            # Resize for consistent processing
            pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
            cv_image = cv2.resize(cv_image, (256, 256))
            
            return pil_image, cv_image
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None
    
    def _extract_phash(self, pil_image):
        """Extract 64-bit perceptual hash"""
        try:
            phash = imagehash.phash(pil_image, hash_size=8)
            # Convert hex string to binary array
            hex_str = str(phash)
            # Convert each hex character to 4 bits, then flatten
            binary_bits = []
            for hex_char in hex_str:
                # Convert hex character to integer, then to 4-bit binary
                bits = format(int(hex_char, 16), '04b')
                binary_bits.extend([int(bit) for bit in bits])
            return np.array(binary_bits, dtype=np.float32)
        except Exception as e:
            print(f"Error extracting pHash: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def _create_white_mask(self, cv_image):
        """Create mask to exclude white background"""
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for non-white pixels
        mask = gray < self.white_threshold
        
        # Apply morphological operations to clean mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _extract_color_histogram(self, cv_image):
        """Extract RGB color histogram with white background masking"""
        try:
            # Create mask to exclude white background
            mask = self._create_white_mask(cv_image)
            
            # If mask is empty, use full image
            if np.sum(mask) == 0:
                mask = None
            
            # Calculate histogram for each channel
            hist_r = cv2.calcHist([cv_image], [2], mask, [32], [0, 256])
            hist_g = cv2.calcHist([cv_image], [1], mask, [32], [0, 256])
            hist_b = cv2.calcHist([cv_image], [0], mask, [32], [0, 256])
            
            # Concatenate and normalize
            histogram = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
            histogram = histogram / (np.sum(histogram) + 1e-8)  # Normalize
            
            return histogram.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting histogram: {e}")
            return np.zeros(96, dtype=np.float32)
    
    def _extract_hu_moments(self, cv_image):
        """Extract 7D Hu moments for shape description"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Calculate moments
            moments = cv2.moments(binary)
            
            # Calculate Hu moments
            hu_moments = cv2.HuMoments(moments)
            
            # Log transform for better numerical stability
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
            
            return hu_moments.flatten().astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting Hu moments: {e}")
            return np.zeros(7, dtype=np.float32)
    
    def _extract_orb_descriptors(self, cv_image):
        """Extract ORB descriptors for local features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None:
                return descriptors
            else:
                return np.array([])
                
        except Exception as e:
            print(f"Error extracting ORB descriptors: {e}")
            return np.array([])
    
    def extract_features_single(self, image_path):
        """Extract all features for a single image"""
        # Load image
        pil_image, cv_image = self._load_image(image_path)
        if pil_image is None or cv_image is None:
            return None, None
        
        # Extract features
        phash = self._extract_phash(pil_image)
        histogram = self._extract_color_histogram(cv_image)
        hu_moments = self._extract_hu_moments(cv_image)
        orb_descriptors = self._extract_orb_descriptors(cv_image)
        
        # Combine features (exclude ORB from main vector)
        combined_features = np.concatenate([phash, histogram, hu_moments])
        
        return combined_features, orb_descriptors
    
    def extract_all_features(self, output_dir="./"):
        """Extract features for all images in logo directory"""
        print("Starting feature extraction...")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.svg']:
            import glob
            image_files.extend(glob.glob(os.path.join(self.logo_dir, ext)))
            image_files.extend(glob.glob(os.path.join(self.logo_dir, ext.upper())))
        
        print(f"Found {len(image_files)} images")
        
        # Extract features
        for image_path in tqdm(image_files, desc="Extracting features"):
            filename = os.path.basename(image_path)
            
            features, orb_desc = self.extract_features_single(image_path)
            
            if features is not None:
                self.features[filename] = features
                if len(orb_desc) > 0:
                    self.orb_descriptors[filename] = orb_desc
        
        print(f"Extracted features for {len(self.features)} images")
        
        # Save features
        features_path = os.path.join(output_dir, "features.pkl")
        orb_path = os.path.join(output_dir, "orb_descriptors.pkl")
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.features, f)
        
        with open(orb_path, 'wb') as f:
            pickle.dump(self.orb_descriptors, f)
        
        print(f"Features saved to {features_path}")
        print(f"ORB descriptors saved to {orb_path}")
        
        return self.features, self.orb_descriptors
    
    def load_features(self, features_path, orb_path):
        """Load previously extracted features"""
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        
        with open(orb_path, 'rb') as f:
            self.orb_descriptors = pickle.load(f)
        
        print(f"Loaded features for {len(self.features)} images")
        return self.features, self.orb_descriptors


def main():
    """Main function for feature extraction"""
    logo_dir = "../logos"
    output_dir = "../"
    
    extractor = LogoFeatureExtractor(logo_dir)
    features, orb_descriptors = extractor.extract_all_features(output_dir)
    
    print("Feature extraction complete!")
    print(f"Feature vector dimension: {list(features.values())[0].shape[0]}")
    print(f"Images with ORB descriptors: {len(orb_descriptors)}")


if __name__ == "__main__":
    main()

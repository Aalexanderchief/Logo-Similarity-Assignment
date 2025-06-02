"""
Logo Search Module
Provides similarity search with ORB refinement
"""

import os
import cv2
import numpy as np
import pickle
from features import LogoFeatureExtractor
from indexer import LogoIndexer

class LogoSearcher:
    def __init__(self, logo_dir, index_path, filenames_path, scaler_path, orb_path):
        """
        Initialize logo searcher
        
        Args:
            logo_dir (str): Directory containing logo images
            index_path (str): Path to FAISS index
            filenames_path (str): Path to filenames pickle
            scaler_path (str): Path to feature scaler
            orb_path (str): Path to ORB descriptors pickle
        """
        self.logo_dir = logo_dir
        self.indexer = LogoIndexer()
        self.feature_extractor = LogoFeatureExtractor(logo_dir)
        
        # Load index and ORB descriptors
        self.indexer.load_index(index_path, filenames_path, scaler_path)
        self.load_orb_descriptors(orb_path)
        
        # Initialize ORB matcher
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def load_orb_descriptors(self, orb_path):
        """Load ORB descriptors"""
        with open(orb_path, 'rb') as f:
            self.orb_descriptors = pickle.load(f)
        print(f"Loaded ORB descriptors for {len(self.orb_descriptors)} images")
    
    def _calculate_orb_similarity(self, desc1, desc2, ratio_threshold=0.75):
        """
        Calculate ORB similarity between two descriptor sets
        
        Args:
            desc1, desc2: ORB descriptor arrays
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            int: Number of good matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        
        try:
            # Match descriptors
            matches = self.orb_matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = 0
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches += 1
            
            return good_matches
            
        except Exception as e:
            # Handle case where k=2 matches not available
            try:
                matches = self.orb_matcher.match(desc1, desc2)
                # Use distance threshold instead
                good_matches = sum(1 for m in matches if m.distance < 50)
                return good_matches
            except:
                return 0
    
    def search_similar_logos(self, query_image_path, k=20, orb_refine=True):
        """
        Search for similar logos with optional ORB refinement
        
        Args:
            query_image_path (str): Path to query image
            k (int): Number of candidates to retrieve from FAISS
            orb_refine (bool): Whether to refine results using ORB matching
            
        Returns:
            list: List of (filename, faiss_distance, orb_matches) tuples, sorted by similarity
        """
        # Extract features for query image
        query_features, query_orb = self.feature_extractor.extract_features_single(query_image_path)
        
        if query_features is None:
            print(f"Failed to extract features from {query_image_path}")
            return []
        
        # Get initial candidates from FAISS
        candidates = self.indexer.search_similar(query_features, k=k)
        
        if not orb_refine:
            return [(filename, dist, 0) for filename, dist in candidates]
        
        # Refine with ORB matching
        refined_results = []
        
        for filename, faiss_dist in candidates:
            orb_matches = 0
            
            # Get ORB descriptors for candidate
            if filename in self.orb_descriptors and query_orb is not None and len(query_orb) > 0:
                candidate_orb = self.orb_descriptors[filename]
                orb_matches = self._calculate_orb_similarity(query_orb, candidate_orb)
            
            refined_results.append((filename, faiss_dist, orb_matches))
        
        # Sort by combined score: prioritize ORB matches, then FAISS distance
        refined_results.sort(key=lambda x: (-x[2], x[1]))
        
        return refined_results
    
    def search_by_filename(self, query_filename, k=10, orb_refine=True):
        """
        Search for similar logos using a filename from the dataset
        
        Args:
            query_filename (str): Filename of query image
            k (int): Number of similar images to return
            orb_refine (bool): Whether to refine results using ORB matching
            
        Returns:
            list: List of (filename, faiss_distance, orb_matches) tuples
        """
        query_path = os.path.join(self.logo_dir, query_filename)
        
        if not os.path.exists(query_path):
            print(f"Query file not found: {query_path}")
            return []
        
        return self.search_similar_logos(query_path, k=k+1, orb_refine=orb_refine)[1:]  # Exclude self
    
    def batch_search(self, filenames, k=10, orb_refine=True):
        """
        Perform batch search for multiple query images
        
        Args:
            filenames (list): List of query filenames
            k (int): Number of similar images per query
            orb_refine (bool): Whether to refine results using ORB matching
            
        Returns:
            dict: Dictionary mapping query filename to results list
        """
        results = {}
        
        print(f"Performing batch search for {len(filenames)} queries...")
        
        for filename in filenames:
            try:
                results[filename] = self.search_by_filename(filename, k=k, orb_refine=orb_refine)
            except Exception as e:
                print(f"Error searching for {filename}: {e}")
                results[filename] = []
        
        return results
    
    def get_similarity_matrix(self, filenames, use_orb=True):
        """
        Compute similarity matrix for given filenames
        
        Args:
            filenames (list): List of filenames to compute similarities for
            use_orb (bool): Whether to include ORB similarity
            
        Returns:
            np.array: Similarity matrix
        """
        n = len(filenames)
        similarity_matrix = np.zeros((n, n))
        
        for i, fname1 in enumerate(filenames):
            for j, fname2 in enumerate(filenames):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:  # Compute only upper triangle
                    # Get FAISS distance
                    features1 = self.indexer.get_feature_vector(fname1)
                    features2 = self.indexer.get_feature_vector(fname2)
                    
                    if features1 is not None and features2 is not None:
                        # Normalize and compute distance
                        feat1_norm = self.indexer.scaler.transform(features1.reshape(1, -1))
                        feat2_norm = self.indexer.scaler.transform(features2.reshape(1, -1))
                        faiss_dist = np.linalg.norm(feat1_norm - feat2_norm)
                        
                        # Add ORB similarity if requested
                        orb_sim = 0
                        if use_orb and fname1 in self.orb_descriptors and fname2 in self.orb_descriptors:
                            orb_matches = self._calculate_orb_similarity(
                                self.orb_descriptors[fname1], 
                                self.orb_descriptors[fname2]
                            )
                            orb_sim = orb_matches / 100.0  # Normalize
                        
                        # Combined similarity (lower FAISS distance + higher ORB = higher similarity)
                        combined_sim = max(0, 1 - faiss_dist/10) + orb_sim
                        similarity_matrix[i, j] = combined_sim
                        similarity_matrix[j, i] = combined_sim  # Symmetric
        
        return similarity_matrix


def main():
    """Test the search functionality"""
    logo_dir = "../logos"
    index_path = "../logo_index.faiss"
    filenames_path = "../index_filenames.pkl"
    scaler_path = "../feature_scaler.pkl"
    orb_path = "../orb_descriptors.pkl"
    
    # Initialize searcher
    searcher = LogoSearcher(logo_dir, index_path, filenames_path, scaler_path, orb_path)
    
    # Test search
    test_filename = "aamco-bellevue_com.jpg"
    print(f"Searching for similar logos to: {test_filename}")
    
    results = searcher.search_by_filename(test_filename, k=10, orb_refine=True)
    
    print("Results:")
    for i, (filename, faiss_dist, orb_matches) in enumerate(results):
        print(f"{i+1:2d}. {filename:<50} | FAISS: {faiss_dist:.3f} | ORB: {orb_matches:3d}")


if __name__ == "__main__":
    main()

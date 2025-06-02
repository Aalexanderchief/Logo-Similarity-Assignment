"""
Logo Indexing Module
Creates FAISS index for fast similarity search
"""

import os
import numpy as np
import faiss
import pickle
from sklearn.preprocessing import StandardScaler
import joblib

class LogoIndexer:
    def __init__(self):
        """Initialize the logo indexer"""
        self.index = None
        self.filenames = []
        self.features = {}
        self.scaler = StandardScaler()
        self.feature_matrix = None
        
    def load_features(self, features_path):
        """Load features from pickle file"""
        with open(features_path, 'rb') as f:
            self.features = pickle.load(f)
        
        print(f"Loaded features for {len(self.features)} images")
        return self.features
    
    def _normalize_features(self, features_dict):
        """Normalize features using StandardScaler, filtering out inconsistent shapes"""
        filenames = list(features_dict.keys())
        
        # Check feature shapes and filter out inconsistent ones
        feature_shapes = {}
        for fn in filenames:
            shape = np.array(features_dict[fn]).shape
            if shape not in feature_shapes:
                feature_shapes[shape] = 0
            feature_shapes[shape] += 1
        
        print(f"Feature shape distribution: {feature_shapes}")
        
        # Use the most common shape
        target_shape = max(feature_shapes.keys(), key=lambda x: feature_shapes[x])
        print(f"Using target shape: {target_shape} ({feature_shapes[target_shape]} files)")
        
        # Filter features to only include those with the target shape
        valid_filenames = []
        valid_features = []
        
        for fn in filenames:
            feature_array = np.array(features_dict[fn])
            if feature_array.shape == target_shape:
                valid_filenames.append(fn)
                valid_features.append(feature_array)
            else:
                print(f"Skipping {fn} with shape {feature_array.shape}")
        
        print(f"Using {len(valid_features)} out of {len(filenames)} files for indexing")
        
        feature_matrix = np.array(valid_features)
        
        # Fit scaler and transform
        normalized_matrix = self.scaler.fit_transform(feature_matrix)
        
        return normalized_matrix, valid_filenames
    
    def build_index(self, features_path, output_dir="./"):
        """Build FAISS index from features"""
        print("Building FAISS index...")
        
        # Load features
        self.load_features(features_path)
        
        # Normalize features
        self.feature_matrix, self.filenames = self._normalize_features(self.features)
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        # Create FAISS index
        dimension = self.feature_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        self.index.add(self.feature_matrix.astype(np.float32))
        
        print(f"Index built with {self.index.ntotal} vectors")
        
        # Save index and metadata
        index_path = os.path.join(output_dir, "logo_index.faiss")
        filenames_path = os.path.join(output_dir, "index_filenames.pkl")
        scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
        
        faiss.write_index(self.index, index_path)
        
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Index saved to {index_path}")
        print(f"Filenames saved to {filenames_path}")
        print(f"Scaler saved to {scaler_path}")
        
        return self.index
    
    def load_index(self, index_path, filenames_path, scaler_path):
        """Load pre-built FAISS index"""
        self.index = faiss.read_index(index_path)
        
        with open(filenames_path, 'rb') as f:
            self.filenames = pickle.load(f)
        
        self.scaler = joblib.load(scaler_path)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        return self.index
    
    def search_similar(self, query_features, k=10):
        """
        Search for similar logos using FAISS index
        
        Args:
            query_features (np.array): Feature vector of query image
            k (int): Number of similar images to return
            
        Returns:
            list: List of (filename, distance) tuples
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        # Normalize query features
        query_normalized = self.scaler.transform(query_features.reshape(1, -1))
        
        # Search
        distances, indices = self.index.search(query_normalized.astype(np.float32), k)
        
        # Convert to list of tuples
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid index
                filename = self.filenames[idx]
                results.append((filename, float(dist)))
        
        return results
    
    def get_feature_vector(self, filename):
        """Get feature vector for a specific filename"""
        if filename in self.features:
            return self.features[filename]
        return None
    
    def get_statistics(self):
        """Get index statistics"""
        if self.index is None:
            return None
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'filenames_count': len(self.filenames),
            'feature_range': {
                'min': float(np.min(self.feature_matrix)) if self.feature_matrix is not None else None,
                'max': float(np.max(self.feature_matrix)) if self.feature_matrix is not None else None,
                'mean': float(np.mean(self.feature_matrix)) if self.feature_matrix is not None else None,
            }
        }
        
        return stats


def main():
    """Main function for indexing"""
    features_path = "../features.pkl"
    output_dir = "../"
    
    indexer = LogoIndexer()
    index = indexer.build_index(features_path, output_dir)
    
    # Print statistics
    stats = indexer.get_statistics()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("Indexing complete!")


if __name__ == "__main__":
    main()

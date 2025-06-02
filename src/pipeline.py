"""
Main Pipeline Runner
Orchestrates the complete logo similarity pipeline
"""

import os
import time
import argparse
from features import LogoFeatureExtractor
from indexer import LogoIndexer
from search import LogoSearcher
from grouping import LogoGrouper
from utils import create_website_mapping, create_thumbnails, validate_image_files

def run_pipeline(
    logo_dir="../logos",
    output_dir="../",
    results_dir="../results",
    thumbnails_dir="../thumbnails",
    skip_features=False,
    skip_indexing=False,
    skip_grouping=False,
    validate_images=True,
    create_thumbs=True
):
    """
    Run the complete logo similarity pipeline
    
    Args:
        logo_dir (str): Directory containing logo images
        output_dir (str): Directory to save features and index files
        results_dir (str): Directory to save grouping results
        thumbnails_dir (str): Directory to save thumbnails
        skip_features (bool): Skip feature extraction
        skip_indexing (bool): Skip index building
        skip_grouping (bool): Skip grouping
        validate_images (bool): Validate image files
        create_thumbs (bool): Create thumbnail images
    """
    print("=" * 60)
    print("LOGO SIMILARITY PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if create_thumbs:
        os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Step 0: Validate images (optional)
    if validate_images:
        print("\n📋 Step 0: Validating image files...")
        step_start = time.time()
        
        validation = validate_image_files(logo_dir)
        print(f"Valid images: {validation['valid_images']}/{validation['total_files']}")
        print(f"Invalid images: {len(validation['invalid_images'])}")
        print(f"File formats: {validation['formats']}")
        
        if validation['invalid_images']:
            print("Invalid files:", validation['invalid_images'][:5])  # Show first 5
        
        print(f"✅ Validation complete ({time.time() - step_start:.2f}s)")
    
    # Step 1: Feature Extraction
    features_path = os.path.join(output_dir, "features.pkl")
    orb_path = os.path.join(output_dir, "orb_descriptors.pkl")
    
    if not skip_features or not os.path.exists(features_path):
        print("\n🔍 Step 1: Extracting features...")
        step_start = time.time()
        
        extractor = LogoFeatureExtractor(logo_dir)
        features, orb_descriptors = extractor.extract_all_features(output_dir)
        
        print(f"✅ Feature extraction complete ({time.time() - step_start:.2f}s)")
        print(f"   Features extracted for {len(features)} images")
        print(f"   ORB descriptors for {len(orb_descriptors)} images")
    else:
        print("\n⏭️  Step 1: Skipping feature extraction (files exist)")
    
    # Step 2: Index Building
    index_path = os.path.join(output_dir, "logo_index.faiss")
    filenames_path = os.path.join(output_dir, "index_filenames.pkl")
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    
    if not skip_indexing or not os.path.exists(index_path):
        print("\n🏗️  Step 2: Building FAISS index...")
        step_start = time.time()
        
        indexer = LogoIndexer()
        index = indexer.build_index(features_path, output_dir)
        
        # Print statistics
        stats = indexer.get_statistics()
        print(f"✅ Index building complete ({time.time() - step_start:.2f}s)")
        print(f"   Index size: {stats['total_vectors']} vectors")
        print(f"   Dimensions: {stats['dimension']}")
    else:
        print("\n⏭️  Step 2: Skipping index building (files exist)")
    
    # Step 3: Logo Grouping
    summary_path = os.path.join(results_dir, "summary.json")
    
    if not skip_grouping or not os.path.exists(summary_path):
        print("\n👥 Step 3: Grouping logos...")
        step_start = time.time()
        
        # Initialize searcher and grouper
        searcher = LogoSearcher(logo_dir, index_path, filenames_path, scaler_path, orb_path)
        grouper = LogoGrouper(searcher, results_dir)
        
        # Group logos
        results = grouper.group_logos(k_neighbors=20)
        grouper.save_results()
        
        # Print statistics
        stats = grouper.get_group_statistics()
        print(f"✅ Logo grouping complete ({time.time() - step_start:.2f}s)")
        print(f"   Total groups: {stats['total_groups']}")
        print(f"   Tight groups: {stats['tight_groups']}")
        print(f"   Loose groups: {stats['loose_groups']}")
        print(f"   Outliers: {stats['total_outliers']}")
        print(f"   Average group size: {stats['average_group_size']:.1f}")
    else:
        print("\n⏭️  Step 3: Skipping grouping (files exist)")
    
    # Step 4: Create utilities (optional)
    if create_thumbs:
        print("\n🖼️  Step 4: Creating thumbnails...")
        step_start = time.time()
        
        create_thumbnails(logo_dir, thumbnails_dir, size=(128, 128))
        
        print(f"✅ Thumbnails created ({time.time() - step_start:.2f}s)")
    
    # Step 5: Create website mapping
    print("\n🌐 Step 5: Creating website mapping...")
    step_start = time.time()
    
    mapping_path = os.path.join(output_dir, "mapping.csv")
    create_website_mapping(logo_dir, mapping_path)
    
    print(f"✅ Website mapping created ({time.time() - step_start:.2f}s)")
    
    # Pipeline complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("\nOutput files:")
    print(f"  📁 Features: {features_path}")
    print(f"  📁 Index: {index_path}")
    print(f"  📁 Results: {results_dir}/")
    print(f"  📁 Thumbnails: {thumbnails_dir}/")
    print(f"  📁 Mapping: {mapping_path}")
    print("\nTo launch the web interface:")
    print(f"  cd {os.path.dirname(__file__)}")
    print("  streamlit run main.py")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Logo Similarity Pipeline")
    
    parser.add_argument("--logo-dir", default="../logos", 
                       help="Directory containing logo images")
    parser.add_argument("--output-dir", default="../", 
                       help="Directory to save features and index files")
    parser.add_argument("--results-dir", default="../results", 
                       help="Directory to save grouping results")
    parser.add_argument("--thumbnails-dir", default="../thumbnails", 
                       help="Directory to save thumbnails")
    
    parser.add_argument("--skip-features", action="store_true", 
                       help="Skip feature extraction")
    parser.add_argument("--skip-indexing", action="store_true", 
                       help="Skip index building")
    parser.add_argument("--skip-grouping", action="store_true", 
                       help="Skip grouping")
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip image validation")
    parser.add_argument("--skip-thumbnails", action="store_true", 
                       help="Skip thumbnail creation")
    
    # Grouping thresholds
    parser.add_argument("--t1-distance", type=float, default=2.0,
                       help="Tight group FAISS distance threshold")
    parser.add_argument("--m1-orb", type=int, default=10,
                       help="Tight group ORB matches threshold")
    parser.add_argument("--t2-distance", type=float, default=4.0,
                       help="Loose group FAISS distance threshold")
    parser.add_argument("--m2-orb", type=int, default=5,
                       help="Loose group ORB matches threshold")
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        logo_dir=args.logo_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        thumbnails_dir=args.thumbnails_dir,
        skip_features=args.skip_features,
        skip_indexing=args.skip_indexing,
        skip_grouping=args.skip_grouping,
        validate_images=not args.skip_validation,
        create_thumbs=not args.skip_thumbnails
    )

if __name__ == "__main__":
    main()

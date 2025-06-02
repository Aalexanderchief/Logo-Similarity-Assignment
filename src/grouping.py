"""
Logo Grouping Module
Groups similar logos using threshold-based clustering
"""

import os
import numpy as np
import pickle
from collections import defaultdict
from search import LogoSearcher
import json

class LogoGrouper:
    def __init__(self, searcher, output_dir="../results"):
        """
        Initialize logo grouper
        
        Args:
            searcher (LogoSearcher): Initialized logo searcher
            output_dir (str): Directory to save results
        """
        self.searcher = searcher
        self.output_dir = output_dir
        self.groups = {}
        self.outliers = []
        
        # Thresholds for grouping
        self.T1_distance = 2.0      # Tight group threshold (FAISS distance)
        self.M1_orb_matches = 10    # Tight group ORB matches
        self.T2_distance = 4.0      # Loose group threshold  
        self.M2_orb_matches = 5     # Loose group ORB matches
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "groups"), exist_ok=True)
    
    def set_thresholds(self, t1_dist=2.0, m1_orb=10, t2_dist=4.0, m2_orb=5):
        """Set grouping thresholds"""
        self.T1_distance = t1_dist
        self.M1_orb_matches = m1_orb
        self.T2_distance = t2_dist
        self.M2_orb_matches = m2_orb
        
        print(f"Thresholds set: T1={t1_dist}, M1={m1_orb}, T2={t2_dist}, M2={m2_orb}")
    
    def _meets_tight_criteria(self, faiss_dist, orb_matches):
        """Check if similarity meets tight grouping criteria"""
        return faiss_dist < self.T1_distance and orb_matches > self.M1_orb_matches
    
    def _meets_loose_criteria(self, faiss_dist, orb_matches):
        """Check if similarity meets loose grouping criteria"""
        return faiss_dist < self.T2_distance or orb_matches > self.M2_orb_matches
    
    def group_logos(self, k_neighbors=20):
        """
        Group logos using threshold-based clustering
        
        Args:
            k_neighbors (int): Number of neighbors to consider for each logo
            
        Returns:
            dict: Dictionary containing groups and outliers
        """
        print("Starting logo grouping...")
        
        # Get all filenames
        all_filenames = self.searcher.indexer.filenames.copy()
        remaining_logos = set(all_filenames)
        
        self.groups = {}
        group_id = 1
        
        print(f"Processing {len(all_filenames)} logos...")
        
        # Process each logo
        for i, query_filename in enumerate(all_filenames):
            if i % 100 == 0:
                print(f"Processed {i}/{len(all_filenames)} logos, {len(remaining_logos)} remaining")
            
            # Skip if already grouped
            if query_filename not in remaining_logos:
                continue
            
            # Find similar logos
            similar_logos = self.searcher.search_by_filename(
                query_filename, k=k_neighbors, orb_refine=True
            )
            
            # Find tight matches (Group 1 criteria)
            tight_matches = [query_filename]  # Include query itself
            
            for filename, faiss_dist, orb_matches in similar_logos:
                if (filename in remaining_logos and 
                    self._meets_tight_criteria(faiss_dist, orb_matches)):
                    tight_matches.append(filename)
            
            # If we have tight matches, create a group
            if len(tight_matches) > 1:
                group_name = f"group_{group_id}"
                self.groups[group_name] = {
                    'type': 'tight',
                    'members': tight_matches.copy(),
                    'representative': query_filename,
                    'criteria': f"FAISS < {self.T1_distance} AND ORB > {self.M1_orb_matches}"
                }
                
                # Remove from remaining
                for member in tight_matches:
                    remaining_logos.discard(member)
                
                group_id += 1
                continue
            
            # Check for loose matches (Group 2+ criteria)
            loose_matches = [query_filename]
            
            for filename, faiss_dist, orb_matches in similar_logos:
                if (filename in remaining_logos and 
                    self._meets_loose_criteria(faiss_dist, orb_matches)):
                    loose_matches.append(filename)
            
            # If we have loose matches, create a group
            if len(loose_matches) > 1:
                group_name = f"group_{group_id}"
                self.groups[group_name] = {
                    'type': 'loose',
                    'members': loose_matches.copy(),
                    'representative': query_filename,
                    'criteria': f"FAISS < {self.T2_distance} OR ORB > {self.M2_orb_matches}"
                }
                
                # Remove from remaining
                for member in loose_matches:
                    remaining_logos.discard(member)
                
                group_id += 1
        
        # Remaining logos are outliers
        self.outliers = list(remaining_logos)
        
        print(f"Grouping complete!")
        print(f"Created {len(self.groups)} groups")
        print(f"Found {len(self.outliers)} outliers")
        
        return {
            'groups': self.groups,
            'outliers': self.outliers
        }
    
    def save_results(self, include_website_mapping=True):
        """Save grouping results to files"""
        print("Saving results...")
        
        # Load website mapping if available
        website_mapping = {}
        if include_website_mapping:
            website_mapping = self._load_website_mapping()
        
        # Save individual group files
        for group_name, group_data in self.groups.items():
            group_file = os.path.join(self.output_dir, "groups", f"{group_name}.txt")
            
            with open(group_file, 'w') as f:
                f.write(f"# {group_name.replace('_', ' ').title()}\n")
                f.write(f"# Type: {group_data['type']}\n")
                f.write(f"# Criteria: {group_data['criteria']}\n")
                f.write(f"# Representative: {group_data['representative']}\n")
                f.write(f"# Members: {len(group_data['members'])}\n\n")
                
                for member in group_data['members']:
                    website = website_mapping.get(member, self._filename_to_website(member))
                    f.write(f"{member} | {website}\n")
        
        # Save outliers
        outliers_file = os.path.join(self.output_dir, "outliers.txt")
        with open(outliers_file, 'w') as f:
            f.write(f"# Outliers ({len(self.outliers)} logos)\n")
            f.write(f"# These logos don't match any similarity criteria\n\n")
            
            for outlier in self.outliers:
                website = website_mapping.get(outlier, self._filename_to_website(outlier))
                f.write(f"{outlier} | {website}\n")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        summary = {
            'total_logos': len(self.searcher.indexer.filenames),
            'total_groups': len(self.groups),
            'total_outliers': len(self.outliers),
            'thresholds': {
                'T1_distance': self.T1_distance,
                'M1_orb_matches': self.M1_orb_matches,
                'T2_distance': self.T2_distance,
                'M2_orb_matches': self.M2_orb_matches
            },
            'group_stats': {}
        }
        
        for group_name, group_data in self.groups.items():
            summary['group_stats'][group_name] = {
                'type': group_data['type'],
                'size': len(group_data['members']),
                'representative': group_data['representative']
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
        print(f"Groups: {len(self.groups)} files in groups/")
        print(f"Outliers: {len(self.outliers)} logos in outliers.txt")
        print(f"Summary: summary.json")
    
    def _load_website_mapping(self):
        """Load website mapping from CSV if available"""
        mapping_file = os.path.join(self.output_dir, "..", "mapping.csv")
        website_mapping = {}
        
        try:
            import pandas as pd
            df = pd.read_csv(mapping_file)
            
            # Assume first column is filename, second is website
            if len(df.columns) >= 2:
                for _, row in df.iterrows():
                    filename = str(row.iloc[0])
                    website = str(row.iloc[1])
                    website_mapping[filename] = website
                
                print(f"Loaded website mapping for {len(website_mapping)} entries")
        except Exception as e:
            print(f"Could not load website mapping: {e}")
        
        return website_mapping
    
    def _filename_to_website(self, filename):
        """Convert filename to website URL"""
        # Remove extension and replace underscores with dots
        website = filename.rsplit('.', 1)[0]  # Remove extension
        website = website.replace('_', '.')   # Replace underscores
        
        # Add protocol if not present
        if not website.startswith(('http://', 'https://')):
            website = f"https://{website}"
        
        return website
    
    def get_group_statistics(self):
        """Get detailed statistics about groups"""
        if not self.groups:
            return None
        
        stats = {
            'total_groups': len(self.groups),
            'total_outliers': len(self.outliers),
            'group_sizes': [],
            'tight_groups': 0,
            'loose_groups': 0,
            'largest_group': 0,
            'smallest_group': float('inf'),
            'average_group_size': 0
        }
        
        for group_data in self.groups.values():
            size = len(group_data['members'])
            stats['group_sizes'].append(size)
            
            if group_data['type'] == 'tight':
                stats['tight_groups'] += 1
            else:
                stats['loose_groups'] += 1
            
            stats['largest_group'] = max(stats['largest_group'], size)
            stats['smallest_group'] = min(stats['smallest_group'], size)
        
        if stats['group_sizes']:
            stats['average_group_size'] = np.mean(stats['group_sizes'])
        
        return stats
    
    def load_results(self, summary_file=None):
        """Load previously saved results"""
        if summary_file is None:
            summary_file = os.path.join(self.output_dir, "summary.json")
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Load groups
            self.groups = {}
            for group_name in summary['group_stats'].keys():
                group_file = os.path.join(self.output_dir, "groups", f"{group_name}.txt")
                if os.path.exists(group_file):
                    members = []
                    with open(group_file, 'r') as f:
                        for line in f:
                            if not line.startswith('#') and '|' in line:
                                filename = line.split('|')[0].strip()
                                members.append(filename)
                    
                    self.groups[group_name] = {
                        'type': summary['group_stats'][group_name]['type'],
                        'members': members,
                        'representative': summary['group_stats'][group_name]['representative']
                    }
            
            # Load outliers
            outliers_file = os.path.join(self.output_dir, "outliers.txt")
            self.outliers = []
            if os.path.exists(outliers_file):
                with open(outliers_file, 'r') as f:
                    for line in f:
                        if not line.startswith('#') and '|' in line:
                            filename = line.split('|')[0].strip()
                            self.outliers.append(filename)
            
            print(f"Loaded {len(self.groups)} groups and {len(self.outliers)} outliers")
            return True
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False


def main():
    """Main function for grouping"""
    logo_dir = "../logos"
    index_path = "../logo_index.faiss"
    filenames_path = "../index_filenames.pkl"
    scaler_path = "../feature_scaler.pkl"
    orb_path = "../orb_descriptors.pkl"
    
    # Initialize searcher
    print("Initializing searcher...")
    searcher = LogoSearcher(logo_dir, index_path, filenames_path, scaler_path, orb_path)
    
    # Initialize grouper
    grouper = LogoGrouper(searcher)
    
    # Set custom thresholds if needed
    # grouper.set_thresholds(t1_dist=1.5, m1_orb=15, t2_dist=3.0, m2_orb=8)
    
    # Group logos
    results = grouper.group_logos(k_neighbors=20)
    
    # Save results
    grouper.save_results()
    
    # Print statistics
    stats = grouper.get_group_statistics()
    print("\nGrouping Statistics:")
    for key, value in stats.items():
        if key != 'group_sizes':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

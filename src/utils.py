"""
Utility functions for logo similarity pipeline
"""

import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
import json

def filename_to_website(filename):
    """
    Convert filename to website URL
    
    Args:
        filename (str): Logo filename
        
    Returns:
        str: Website URL
    """
    # Remove extension and replace underscores with dots
    website = filename.rsplit('.', 1)[0]  # Remove extension
    website = website.replace('_', '.')   # Replace underscores
    
    # Add protocol if not present
    if not website.startswith(('http://', 'https://')):
        website = f"https://{website}"
    
    return website

def create_website_mapping(logos_dir, output_file="mapping.csv"):
    """
    Create mapping CSV from logo filenames
    
    Args:
        logos_dir (str): Directory containing logo files
        output_file (str): Output CSV file path
    """
    # Get all logo files
    logo_files = []
    for ext in ['jpg', 'jpeg', 'png', 'svg']:
        import glob
        logo_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext}")))
        logo_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext.upper()}")))
    
    # Create mapping
    mapping_data = []
    for logo_path in logo_files:
        filename = os.path.basename(logo_path)
        website = filename_to_website(filename)
        mapping_data.append({
            'filename': filename,
            'website': website,
            'filepath': logo_path
        })
    
    # Save to CSV
    df = pd.DataFrame(mapping_data)
    df.to_csv(output_file, index=False)
    
    print(f"Created website mapping for {len(mapping_data)} logos")
    print(f"Saved to {output_file}")
    
    return df

def convert_svg_to_png_batch(logos_dir, output_dir=None, size=(256, 256)):
    """
    Convert all SVG files to PNG
    
    Args:
        logos_dir (str): Directory containing SVG files
        output_dir (str): Output directory (None = same as input)
        size (tuple): Output image size
    """
    if output_dir is None:
        output_dir = logos_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    import glob
    svg_files = glob.glob(os.path.join(logos_dir, "*.svg"))
    svg_files.extend(glob.glob(os.path.join(logos_dir, "*.SVG")))
    
    converted = 0
    for svg_path in svg_files:
        try:
            # Convert SVG to PNG
            png_data = cairosvg.svg2png(
                url=svg_path, 
                output_width=size[0], 
                output_height=size[1]
            )
            
            # Save PNG
            filename = os.path.basename(svg_path)
            png_filename = filename.rsplit('.', 1)[0] + '.png'
            png_path = os.path.join(output_dir, png_filename)
            
            with open(png_path, 'wb') as f:
                f.write(png_data)
            
            converted += 1
            
        except Exception as e:
            print(f"Error converting {svg_path}: {e}")
    
    print(f"Converted {converted} SVG files to PNG")

def create_thumbnails(logos_dir, thumbnails_dir, size=(128, 128)):
    """
    Create thumbnail images for web interface
    
    Args:
        logos_dir (str): Directory containing original logos
        thumbnails_dir (str): Output directory for thumbnails
        size (tuple): Thumbnail size
    """
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'svg']:
        import glob
        image_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext.upper()}")))
    
    created = 0
    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            thumb_filename = filename.rsplit('.', 1)[0] + '_thumb.png'
            thumb_path = os.path.join(thumbnails_dir, thumb_filename)
            
            # Skip if already exists
            if os.path.exists(thumb_path):
                continue
            
            # Load and resize image
            if image_path.lower().endswith('.svg'):
                # Convert SVG to PNG
                png_data = cairosvg.svg2png(
                    url=image_path, 
                    output_width=size[0], 
                    output_height=size[1]
                )
                image = Image.open(BytesIO(png_data))
            else:
                image = Image.open(image_path)
                image = image.resize(size, Image.Resampling.LANCZOS)
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save thumbnail
            image.save(thumb_path, 'PNG', optimize=True)
            created += 1
            
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
    
    print(f"Created {created} thumbnails")

def validate_image_files(logos_dir):
    """
    Validate all image files in directory
    
    Args:
        logos_dir (str): Directory containing images
        
    Returns:
        dict: Validation results
    """
    import glob
    
    results = {
        'total_files': 0,
        'valid_images': 0,
        'invalid_images': [],
        'file_sizes': [],
        'formats': {},
        'corrupted': []
    }
    
    # Get all image files
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'svg']:
        image_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(logos_dir, f"*.{ext.upper()}")))
    
    results['total_files'] = len(image_files)
    
    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path)
            results['file_sizes'].append(file_size)
            
            # Get file format
            ext = filename.split('.')[-1].lower()
            results['formats'][ext] = results['formats'].get(ext, 0) + 1
            
            # Try to load image
            if ext == 'svg':
                # Validate SVG
                with open(image_path, 'r') as f:
                    content = f.read()
                    if '<svg' in content.lower():
                        results['valid_images'] += 1
                    else:
                        results['invalid_images'].append(filename)
            else:
                # Validate raster images
                try:
                    image = Image.open(image_path)
                    image.verify()  # Check if image is corrupted
                    results['valid_images'] += 1
                except:
                    results['corrupted'].append(filename)
                    results['invalid_images'].append(filename)
                    
        except Exception as e:
            results['invalid_images'].append(f"{filename}: {str(e)}")
    
    # Calculate statistics
    if results['file_sizes']:
        results['avg_file_size'] = np.mean(results['file_sizes'])
        results['min_file_size'] = np.min(results['file_sizes'])
        results['max_file_size'] = np.max(results['file_sizes'])
    
    return results

def load_group_results(results_dir):
    """
    Load grouping results for web interface
    
    Args:
        results_dir (str): Directory containing results
        
    Returns:
        dict: Loaded results
    """
    results = {
        'groups': {},
        'outliers': [],
        'summary': {}
    }
    
    # Load summary
    summary_file = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load groups
    groups_dir = os.path.join(results_dir, "groups")
    if os.path.exists(groups_dir):
        for group_file in os.listdir(groups_dir):
            if group_file.endswith('.txt'):
                group_name = group_file[:-4]  # Remove .txt
                group_path = os.path.join(groups_dir, group_file)
                
                members = []
                with open(group_path, 'r') as f:
                    for line in f:
                        if not line.startswith('#') and '|' in line:
                            parts = line.strip().split('|')
                            filename = parts[0].strip()
                            website = parts[1].strip() if len(parts) > 1 else filename_to_website(filename)
                            members.append({
                                'filename': filename,
                                'website': website
                            })
                
                results['groups'][group_name] = members
    
    # Load outliers
    outliers_file = os.path.join(results_dir, "outliers.txt")
    if os.path.exists(outliers_file):
        with open(outliers_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and '|' in line:
                    parts = line.strip().split('|')
                    filename = parts[0].strip()
                    website = parts[1].strip() if len(parts) > 1 else filename_to_website(filename)
                    results['outliers'].append({
                        'filename': filename,
                        'website': website
                    })
    
    return results

def export_results_csv(results_dir, output_file="logo_groups.csv"):
    """
    Export grouping results to CSV
    
    Args:
        results_dir (str): Directory containing results
        output_file (str): Output CSV file
    """
    results = load_group_results(results_dir)
    
    # Prepare data for CSV
    csv_data = []
    
    # Add groups
    for group_name, members in results['groups'].items():
        for member in members:
            csv_data.append({
                'filename': member['filename'],
                'website': member['website'],
                'group': group_name,
                'type': 'group_member'
            })
    
    # Add outliers
    for outlier in results['outliers']:
        csv_data.append({
            'filename': outlier['filename'],
            'website': outlier['website'],
            'group': 'outlier',
            'type': 'outlier'
        })
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    
    print(f"Exported {len(csv_data)} entries to {output_file}")

def main():
    """Utility functions demonstration"""
    logos_dir = "../logos"
    
    print("Logo Similarity Pipeline Utilities")
    print("=" * 40)
    
    # Create website mapping
    print("1. Creating website mapping...")
    mapping_df = create_website_mapping(logos_dir, "../mapping.csv")
    
    # Validate images
    print("\n2. Validating image files...")
    validation = validate_image_files(logos_dir)
    print(f"Valid images: {validation['valid_images']}/{validation['total_files']}")
    print(f"Invalid images: {len(validation['invalid_images'])}")
    print(f"File formats: {validation['formats']}")
    
    # Create thumbnails
    print("\n3. Creating thumbnails...")
    create_thumbnails(logos_dir, "../thumbnails")
    
    print("\nUtilities complete!")

if __name__ == "__main__":
    main()

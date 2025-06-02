"""
Streamlit Web Interface for Logo Similarity Pipeline
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO

# Import our modules
from search import LogoSearcher
from utils import load_group_results, filename_to_website
from features import LogoFeatureExtractor

# Configure page
st.set_page_config(
    page_title="Logo Similarity Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
LOGO_DIR = "../logos"
THUMBNAILS_DIR = "../thumbnails"
RESULTS_DIR = "../results"
INDEX_PATH = "../logo_index.faiss"
FILENAMES_PATH = "../index_filenames.pkl"
SCALER_PATH = "../feature_scaler.pkl"
ORB_PATH = "../orb_descriptors.pkl"

@st.cache_resource
def load_searcher():
    """Load the logo searcher (cached)"""
    try:
        searcher = LogoSearcher(LOGO_DIR, INDEX_PATH, FILENAMES_PATH, SCALER_PATH, ORB_PATH)
        return searcher
    except Exception as e:
        st.error(f"Error loading searcher: {e}")
        return None

@st.cache_data
def load_results():
    """Load grouping results (cached)"""
    try:
        return load_group_results(RESULTS_DIR)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return {'groups': {}, 'outliers': [], 'summary': {}}

def get_image_path(filename, use_thumbnail=True):
    """Get path to image or thumbnail"""
    if use_thumbnail:
        thumb_name = filename.rsplit('.', 1)[0] + '_thumb.png'
        thumb_path = os.path.join(THUMBNAILS_DIR, thumb_name)
        if os.path.exists(thumb_path):
            return thumb_path
    
    return os.path.join(LOGO_DIR, filename)

def display_logo_grid(logos, columns=5, use_thumbnails=True, show_similarity=False):
    """Display logos in a grid layout"""
    if not logos:
        st.write("No logos to display")
        return
    
    # Create columns
    cols = st.columns(columns)
    
    for i, logo_data in enumerate(logos):
        col_idx = i % columns
        
        with cols[col_idx]:
            if isinstance(logo_data, dict):
                filename = logo_data.get('filename', logo_data.get('name', ''))
                website = logo_data.get('website', filename_to_website(filename))
                similarity_info = logo_data.get('similarity', None)
            elif isinstance(logo_data, tuple):
                # For search results: (filename, faiss_dist, orb_matches)
                filename = logo_data[0]
                website = filename_to_website(filename)
                if show_similarity and len(logo_data) >= 3:
                    faiss_dist, orb_matches = logo_data[1], logo_data[2]
                    similarity_info = f"FAISS: {faiss_dist:.2f} | ORB: {orb_matches}"
                else:
                    similarity_info = None
            else:
                # Simple filename
                filename = logo_data
                website = filename_to_website(filename)
                similarity_info = None
            
            # Display image
            image_path = get_image_path(filename, use_thumbnail=use_thumbnails)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=filename, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {filename}: {e}")
            else:
                st.error(f"Image not found: {filename}")
            
            # Display website link
            st.markdown(f"🔗 [{website}]({website})")
            
            # Display similarity info if available
            if similarity_info:
                st.caption(similarity_info)

def download_group_file(group_name, members):
    """Create download button for group file"""
    # Create file content
    content = f"# {group_name.replace('_', ' ').title()}\n"
    content += f"# Members: {len(members)}\n\n"
    
    for member in members:
        if isinstance(member, dict):
            filename = member.get('filename', '')
            website = member.get('website', filename_to_website(filename))
        else:
            filename = member
            website = filename_to_website(filename)
        content += f"{filename} | {website}\n"
    
    # Create download button
    st.download_button(
        label=f"📥 Download {group_name}.txt",
        data=content,
        file_name=f"{group_name}.txt",
        mime="text/plain",
        key=f"download_{group_name}"
    )

def search_tab():
    """Tab for logo similarity search"""
    st.header("🔍 Logo Similarity Search")
    
    # Load searcher
    searcher = load_searcher()
    if searcher is None:
        st.error("Could not load searcher. Please ensure all index files are present.")
        return
    
    search_method = st.radio(
        "Search method:",
        ["Select from dataset", "Upload image"],
        horizontal=True
    )
    
    if search_method == "Select from dataset":
        # Get available filenames
        available_files = searcher.indexer.filenames
        search_filter = st.text_input("🔍 Filter logos by filename:", "zalando")
        
        if search_filter:
            filtered_files = [f for f in available_files if search_filter.lower() in f.lower()]
        else:
            filtered_files = available_files[:100]  # Show first 100 by default
        
        if not filtered_files:
            st.warning("No logos match your filter")
            return
        
        default_index = 0
        zalando_files = [f for f in filtered_files if 'zalando' in f.lower()]
        if zalando_files:
            default_index = filtered_files.index(zalando_files[0])
        
        # Select logo
        selected_logo = st.selectbox(
            "Select a logo:",
            filtered_files,
            index=default_index,
            format_func=lambda x: f"{x} ({filename_to_website(x)})"
        )
        
        if selected_logo:
            # Display selected logo
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Query Logo:")
                image_path = get_image_path(selected_logo, use_thumbnail=False)
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    st.image(image, caption=selected_logo, use_container_width=True)
                st.markdown(f"🔗 [{filename_to_website(selected_logo)}]({filename_to_website(selected_logo)})")
            
            with col2:
                # Search parameters
                k_results = st.slider("Number of results:", 5, 50, 20)
                use_orb = st.checkbox("Use ORB refinement", True)
                
                if st.button("🔍 Find Similar Logos"):
                    with st.spinner("Searching for similar logos..."):
                        results = searcher.search_by_filename(
                            selected_logo, 
                            k=k_results, 
                            orb_refine=use_orb
                        )
                    
                    if results:
                        st.success(f"Found {len(results)} similar logos")
                        
                        # Display results
                        st.subheader("Similar Logos:")
                        display_logo_grid(results, columns=4, show_similarity=True)
                    else:
                        st.warning("No similar logos found")
    
    else:  # Upload image
        uploaded_file = st.file_uploader(
            "Upload a logo image:",
            type=['png', 'jpg', 'jpeg'],
            key="upload_search"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Query Logo:")
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Search parameters
                k_results = st.slider("Number of results:", 5, 50, 20, key="upload_k")
                use_orb = st.checkbox("Use ORB refinement", True, key="upload_orb")
                
                if st.button("🔍 Find Similar Logos", key="upload_search_btn"):
                    with st.spinner("Searching for similar logos..."):
                        # Save uploaded file temporarily
                        temp_path = "/tmp/query_logo.png"
                        image.save(temp_path)
                        
                        # Search
                        results = searcher.search_similar_logos(
                            temp_path, 
                            k=k_results, 
                            orb_refine=use_orb
                        )
                        
                        # Clean up
                        os.remove(temp_path)
                    
                    if results:
                        st.success(f"Found {len(results)} similar logos")
                        
                        # Display results
                        st.subheader("Similar Logos:")
                        display_logo_grid(results, columns=4, show_similarity=True)
                    else:
                        st.warning("No similar logos found")

def groups_tab():
    """Tab for browsing precomputed groups"""
    st.header("👥 Logo Groups")
    
    # Load results
    results = load_results()
    
    if not results['groups']:
        st.warning("No groups found. Please run the grouping pipeline first.")
        return
    
    # Group selection
    group_names = list(results['groups'].keys())
    group_names.sort(key=lambda x: len(results['groups'][x]), reverse=True)  # Sort by size
    
    default_index = 0
    if 'group_122' in group_names:
        default_index = group_names.index('group_122')
    
    selected_group = st.selectbox(
        "Select a group:",
        group_names,
        index=default_index,
        format_func=lambda x: f"{x.replace('_', ' ').title()} ({len(results['groups'][x])} logos)"
    )
    
    if selected_group:
        members = results['groups'][selected_group]
        
        # Group info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Group Size", len(members))
        
        with col2:
            # Get group type from summary if available
            group_type = "Unknown"
            if 'summary' in results and 'group_stats' in results['summary']:
                if selected_group in results['summary']['group_stats']:
                    group_type = results['summary']['group_stats'][selected_group].get('type', 'Unknown')
            st.metric("Type", group_type.title())
        
        with col3:
            download_group_file(selected_group, members)
        
        # Display group members
        st.subheader(f"Members of {selected_group.replace('_', ' ').title()}:")
        display_logo_grid(members, columns=5)

def outliers_tab():
    """Tab for viewing outliers"""
    st.header("🎯 Outlier Logos")
    
    # Load results
    results = load_results()
    
    if not results['outliers']:
        st.warning("No outliers found.")
        return
    
    st.write(f"Found **{len(results['outliers'])}** outlier logos that don't match any similarity criteria.")
    
    # Download button for outliers
    download_group_file("outliers", results['outliers'])
    
    # Display outliers
    display_logo_grid(results['outliers'], columns=6)

def statistics_tab():
    """Tab for pipeline statistics"""
    st.header("📊 Pipeline Statistics")
    
    # Load results
    results = load_results()
    
    if 'summary' in results and results['summary']:
        summary = results['summary']
        
        # Overall stats
        st.subheader("Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Logos", summary.get('total_logos', 0))
        
        with col2:
            st.metric("Total Groups", summary.get('total_groups', 0))
        
        with col3:
            st.metric("Total Outliers", summary.get('total_outliers', 0))
        
        with col4:
            grouped_logos = summary.get('total_logos', 0) - summary.get('total_outliers', 0)
            total_logos = summary.get('total_logos', 1)  # Avoid division by zero
            grouping_rate = (grouped_logos / total_logos) * 100
            st.metric("Grouping Rate", f"{grouping_rate:.1f}%")
        
        # Group statistics
        if 'group_stats' in summary and summary['group_stats']:
            st.subheader("Group Statistics")
            
            group_data = []
            for group_name, stats in summary['group_stats'].items():
                group_data.append({
                    'Group': group_name.replace('_', ' ').title(),
                    'Type': stats.get('type', 'Unknown').title(),
                    'Size': stats.get('size', 0),
                    'Representative': stats.get('representative', 'Unknown')
                })
            
            df = pd.DataFrame(group_data)
            df = df.sort_values('Size', ascending=False)
            
            st.dataframe(df, use_container_width=True)
            
            # Group size distribution
            st.subheader("Group Size Distribution")
            sizes = df['Size'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Largest Group", max(sizes) if sizes else 0)
                st.metric("Smallest Group", min(sizes) if sizes else 0)
            
            with col2:
                st.metric("Average Group Size", f"{np.mean(sizes):.1f}" if sizes else "0")
                st.metric("Median Group Size", f"{np.median(sizes):.1f}" if sizes else "0")
        
        # Thresholds used
        if 'thresholds' in summary:
            st.subheader("Grouping Thresholds")
            thresholds = summary['thresholds']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Tight Groups (Group 1):**")
                st.write(f"- FAISS Distance < {thresholds.get('T1_distance', 'Unknown')}")
                st.write(f"- ORB Matches > {thresholds.get('M1_orb_matches', 'Unknown')}")
            
            with col2:
                st.write("**Loose Groups (Group 2+):**")
                st.write(f"- FAISS Distance < {thresholds.get('T2_distance', 'Unknown')}")
                st.write(f"- ORB Matches > {thresholds.get('M2_orb_matches', 'Unknown')}")
    
    else:
        st.warning("No statistics available. Please run the grouping pipeline first.")

def main():
    """Main Streamlit application"""
    # Title and description
    st.title("🔍 Logo Similarity Pipeline")
    st.markdown("Analyze and group similar company logos using computer vision")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.write("Use the tabs below to:")
        st.write("🔍 **Search** - Find similar logos")
        st.write("👥 **Groups** - Browse logo groups")
        st.write("🎯 **Outliers** - View unique logos")
        st.write("📊 **Statistics** - Pipeline stats")
        
        # File status
        st.header("System Status")
        
        files_to_check = [
            ("Index", INDEX_PATH),
            ("Features", "../features.pkl"),
            ("ORB Descriptors", ORB_PATH),
            ("Results", f"{RESULTS_DIR}/summary.json")
        ]
        
        for name, path in files_to_check:
            if os.path.exists(path):
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "👥 Groups", "🎯 Outliers", "📊 Statistics"])
    
    with tab1:
        search_tab()
    
    with tab2:
        groups_tab()
    
    with tab3:
        outliers_tab()
    
    with tab4:
        statistics_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("Logo Similarity Pipeline - Built with Streamlit, OpenCV, FAISS, and ❤️")

if __name__ == "__main__":
    main()

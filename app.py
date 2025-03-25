import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64
from streamlit_lottie import streamlit_lottie
import requests
import json
from streamlit_option_menu import option_menu
import altair as alt
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stoggle import stoggle
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
from streamlit_extras.grid import grid
from streamlit_extras.chart_container import chart_container
from streamlit_extras.stateful_button import button
import re
import os
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="SoundMatch - Headphone Recommendation System",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #7C3AED;
        --secondary-color: #4F46E5;
        --accent-color: #EC4899;
        --background-color: #F9FAFB;
        --card-bg-color: #FFFFFF;
        --text-color: #1F2937;
        --light-text: #6B7280;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
    }
    
    /* Dark mode colors */
    .dark {
        --primary-color: #8B5CF6;
        --secondary-color: #6366F1;
        --accent-color: #F472B6;
        --background-color: #111827;
        --card-bg-color: #1F2937;
        --text-color: #F9FAFB;
        --light-text: #9CA3AF;
    }
    
    /* Main container styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--secondary-color);
        margin-bottom: 1rem;
    }
    
    /* Card styling with animations */
    .product-card {
        border-radius: 16px;
        padding: 1.5rem;
        background-color: var(--card-bg-color);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
        overflow: hidden;
        position: relative;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }
    
    .product-card:hover .card-overlay {
        opacity: 1;
    }
    
    .card-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(236, 72, 153, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 20px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-primary {
        background-color: var(--primary-color);
        color: white;
    }
    
    .badge-secondary {
        background-color: var(--secondary-color);
        color: white;
    }
    
    .badge-accent {
        background-color: var(--accent-color);
        color: white;
    }
    
    .badge-success {
        background-color: var(--success-color);
        color: white;
    }
    
    .badge-warning {
        background-color: var(--warning-color);
        color: white;
    }
    
    .badge-error {
        background-color: var(--error-color);
        color: white;
    }
    
    /* Price tag styling */
    .price-tag {
        background-color: var(--success-color);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Rating tag styling */
    .rating-tag {
        background-color: var(--warning-color);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Reviews tag styling */
    .reviews-tag {
        background-color: var(--secondary-color);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Button styling */
    .custom-button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Divider styling */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        margin: 2rem 0;
        border-radius: 3px;
    }
    
    /* Animated progress bar */
    .progress-container {
        width: 100%;
        height: 8px;
        background-color: #E5E7EB;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        width: 0;
        transition: width 1s ease;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: var(--text-color);
        color: var(--card-bg-color);
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Feature comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .comparison-table th, .comparison-table td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .comparison-table th {
        background-color: var(--primary-color);
        color: white;
    }
    
    .comparison-table tr:nth-child(even) {
        background-color: rgba(0, 0, 0, 0.02);
    }
    
    /* Animation for cards */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Animation for pulse */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Animation for slide in */
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .animate-slide-in-right {
        animation: slideInRight 0.5s ease forwards;
    }
    
    /* Animation for slide in from left */
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .animate-slide-in-left {
        animation: slideInLeft 0.5s ease forwards;
    }
    
    /* Animation for fade in up */
    @keyframes fadeInUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .animate-fade-in-up {
        animation: fadeInUp 0.5s ease forwards;
    }
    
    /* Product image container */
    .product-image-container {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        margin-bottom: 1rem;
        aspect-ratio: 1 / 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #f8f9fa;
    }
    
    .product-image {
        max-width: 100%;
        max-height: 100%;
        transition: transform 0.3s ease;
    }
    
    .product-card:hover .product-image {
        transform: scale(1.05);
    }
    
    /* Product details */
    .product-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
        line-height: 1.4;
        height: 3em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .product-brand {
        color: var(--light-text);
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .product-meta {
        display: flex;
        justify-content: space-between;
        margin: 0.8rem 0;
    }
    
    .product-description {
        font-size: 0.9rem;
        color: var(--light-text);
        margin: 0.5rem 0;
        height: 3.6em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
    
    /* Product card footer */
    .product-card-footer {
        margin-top: auto;
        padding-top: 1rem;
    }
    
    /* Search bar styling */
    .search-container {
        position: relative;
        margin-bottom: 1.5rem;
    }
    
    .search-input {
        width: 100%;
        padding: 0.8rem 1rem 0.8rem 3rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .search-input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        outline: none;
    }
    
    .search-icon {
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        color: var(--light-text);
    }
    
    /* Filter panel */
    .filter-panel {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    
    .filter-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .filter-section {
        margin-bottom: 1.5rem;
    }
    
    .filter-section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Modal styling */
    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .modal-content {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 2rem;
        max-width: 800px;
        width: 90%;
        max-height: 90vh;
        overflow-y: auto;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .modal-close {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: var(--light-text);
    }
    
    .modal-close:hover {
        color: var(--primary-color);
    }
    
    /* Product detail styling */
    .product-detail-image {
        max-width: 100%;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .product-detail-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .product-detail-brand {
        font-size: 1.2rem;
        color: var(--light-text);
        margin-bottom: 1rem;
    }
    
    .product-detail-meta {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .product-detail-description {
        margin: 1.5rem 0;
        line-height: 1.6;
    }
    
    .product-detail-features {
        margin: 1.5rem 0;
    }
    
    .feature-list {
        list-style-type: none;
        padding: 0;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.5rem;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.02);
        border-radius: 8px;
    }
    
    .feature-icon {
        color: var(--primary-color);
    }
    
    /* Error message styling */
    .error-container {
        text-align: center;
        padding: 3rem;
        background-color: var(--card-bg-color);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    }
    
    .error-icon {
        font-size: 4rem;
        color: var(--error-color);
        margin-bottom: 1rem;
    }
    
    .error-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .error-message {
        color: var(--light-text);
        margin-bottom: 1.5rem;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-left-color: var(--primary-color);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Customize Streamlit components */
    .stSelectbox > div > div {
        background-color: var(--card-bg-color);
        border-radius: 8px;
    }
    
    .stSlider > div > div {
        background-color: var(--primary-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
        }
        .product-grid {
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        }
    }
    
    /* Star rating component */
    .star-rating {
        display: inline-flex;
        align-items: center;
    }
    
    .star {
        color: #F59E0B;
        font-size: 1.2rem;
    }
    
    .star-empty {
        color: #E5E7EB;
    }
    
    /* Availability indicator */
    .availability-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .availability-high {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }
    
    .availability-medium {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
    }
    
    .availability-low {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
    }
    
    /* Loyalty points badge */
    .loyalty-points {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        background-color: rgba(79, 70, 229, 0.1);
        color: var(--secondary-color);
    }
    
    /* Product grid */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1.5rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        background-color: var(--card-bg-color);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    }
    
    .empty-state-icon {
        font-size: 4rem;
        color: var(--light-text);
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .empty-state-message {
        color: var(--light-text);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
headphone_animation = load_lottieurl("https://lottie.host/e9e39b2c-7c0e-4b25-b6e3-b9c5e5a1e8e4/Rl9sBMZZQl.json")
search_animation = load_lottieurl("https://lottie.host/c2a2c6b9-5d5e-4a10-8cc5-b6e3b9c5e5a1/Rl9sBMZZQl.json")
compare_animation = load_lottieurl("https://lottie.host/a1e8e4-7c0e-4b25-b6e3-b9c5e5a1e8e4/Rl9sBMZZQl.json")

# Function to add background image
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add subtle background pattern
add_bg_from_url("https://www.transparenttextures.com/patterns/cubes.png")

# Toggle dark/light mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply dark mode class if enabled
if st.session_state.dark_mode:
    st.markdown("""
    <script>
        document.body.classList.add('dark');
    </script>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <script>
        document.body.classList.remove('dark');
    </script>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

if 'show_product_detail' not in st.session_state:
    st.session_state.show_product_detail = False

if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

if 'filter_applied' not in st.session_state:
    st.session_state.filter_applied = False

# Function to show product detail
def show_product_detail(product_name):
    st.session_state.selected_product = product_name
    st.session_state.show_product_detail = True

# Function to close product detail
def close_product_detail():
    st.session_state.show_product_detail = False

# Function to handle search
def handle_search():
    st.session_state.filter_applied = True

# Function to reset filters
def reset_filters():
    st.session_state.filter_applied = False
    st.session_state.search_query = ""

# Function to extract features from product description
def extract_features(description):
    # Split by commas or periods
    if isinstance(description, str):
        features = re.split(r'[,.]', description)
        # Clean up and filter out empty features
        features = [f.strip() for f in features if f.strip()]
        return features
    return []

# Function to determine availability status
def get_availability_status(availability):
    if availability >= 70:
        return "High", "availability-high"
    elif availability >= 40:
        return "Medium", "availability-medium"
    else:
        return "Low", "availability-low"

# Function to render star rating
def render_star_rating(rating):
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)
    
    stars_html = ""
    for _ in range(full_stars):
        stars_html += '<span class="star">★</span>'
    
    if half_star:
        stars_html += '<span class="star">★</span>'
    
    for _ in range(empty_stars):
        stars_html += '<span class="star star-empty">★</span>'
    
    return f'<div class="star-rating">{stars_html} <span style="margin-left: 0.5rem;">{rating}</span></div>'

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Try to load the CSV file
        df = pd.read_csv('productdata.csv')
        
        # Clean up column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Extract headphone type from name or description
        def extract_type(row):
            name = str(row['name']).lower()
            description = str(row['description']).lower()
            
            if 'over' in name or 'over' in description or 'over-ear' in name or 'over-ear' in description:
                return 'Over-Ear'
            elif 'on' in name or 'on' in description or 'on-ear' in name or 'on-ear' in description:
                return 'On-Ear'
            elif 'in' in name or 'in' in description or 'in-ear' in name or 'in-ear' in description or 'earphone' in name or 'earphone' in description:
                return 'In-Ear'
            else:
                return 'Other'
        
        df['type'] = df.apply(extract_type, axis=1)
        
        # Extract connectivity type
        def extract_connectivity(row):
            name = str(row['name']).lower()
            description = str(row['description']).lower()
            
            if 'wireless' in name or 'wireless' in description or 'bluetooth' in name or 'bluetooth' in description:
                return 'Wireless'
            else:
                return 'Wired'
        
        df['connectivity'] = df.apply(extract_connectivity, axis=1)
        
        # Extract battery life if available
        def extract_battery_life(row):
            name = str(row['name']).lower()
            description = str(row['description']).lower()
            
            # Look for patterns like "XX hours", "XXh", "XX hrs"
            battery_patterns = [
                r'(\d+)\s*hours',
                r'(\d+)\s*hrs',
                r'(\d+)\s*hr',
                r'(\d+)\s*h\b',
                r'(\d+)-hour'
            ]
            
            for pattern in battery_patterns:
                name_match = re.search(pattern, name)
                desc_match = re.search(pattern, description)
                
                if name_match:
                    return int(name_match.group(1))
                elif desc_match:
                    return int(desc_match.group(1))
            
            # Default value based on connectivity
            return 0 if row['connectivity'] == 'Wired' else 20  # Assume 20 hours for wireless if not specified
        
        df['battery_life'] = df.apply(extract_battery_life, axis=1)
        
        # Extract base model name
        df['base_model'] = df['name'].apply(lambda x: x.split('(')[0].strip() if '(' in str(x) else str(x))
        
        # Combine text features into a single column
        df['combined_text'] = df['name'].astype(str) + ' ' + df['brand'].astype(str) + ' ' + df['description'].astype(str) + ' ' + df['category'].astype(str)
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        df[['price_normalized', 'rating_normalized', 'reviews_normalized']] = scaler.fit_transform(df[['price', 'rating', 'reviews']])
        
        # TF-IDF for text features
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        
        # Combine TF-IDF and numerical features
        numerical_features = df[['price_normalized', 'rating_normalized', 'reviews_normalized']].values
        feature_matrix = hstack([tfidf_matrix, numerical_features])
        
        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
        
        return df, cosine_sim
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataset for demonstration
        data = {
            'name': [
                'HAMMER Bash Max Over The Ear Wireless Bluetooth Headphones',
                'boAt Rockerz 450, 15 HRS Battery, 40mm Drivers',
                'boAt Bassheads 100 in Ear Wired Earphones with Mic',
                'Sony WH-CH520 Wireless Bluetooth Headphones with Mic',
                'ZEBRONICS THUNDER Bluetooth 5.3 Wireless Headphones',
                'boAt Bassheads 900 Pro Wired Headphones with 40Mm Drivers',
                'Boult Q Over Ear Bluetooth Headphones with 70H Playtime'
            ],
            'brand': ['Hammer', 'Boat', 'Boat', 'Sony', 'Zebronics', 'Boat', 'Boult'],
            'price': [2299, 1499, 297, 3989, 699, 898, 1799],
            'rating': [3.7, 4.0, 4.1, 4.2, 3.8, 4.2, 4.2],
            'reviews': [3136, 115737, 415342, 17809, 75791, 98203, 1747],
            'category': ['Headphone', 'Headphone', 'Headphone', 'Headphone', 'Headphone', 'Headphone', 'Headphone'],
            'image_url': [
                'https://m.media-amazon.com/images/I/315ZO+wzU7L._SY300_SX300_.jpg',
                'https://m.media-amazon.com/images/I/31DU-7yXUyL._SX300_SY300_QL70_FMwebp_.jpg',
                'https://m.media-amazon.com/images/I/313U7Xx9b4L._SX300_SY300_QL70_FMwebp_.jpg',
                'https://m.media-amazon.com/images/I/318RvHnDwHL._SX300_SY300_QL70_FMwebp_.jpg',
                'https://m.media-amazon.com/images/I/417gW8O1RzL._SX300_SY300_QL70_FMwebp_.jpg',
                'https://m.media-amazon.com/images/I/4192vscwlSL._SX300_SY300_QL70_FMwebp_.jpg',
                'https://m.media-amazon.com/images/I/318EgLiOMUL._SX300_SY300_QL70_FMwebp_.jpg'
            ],
            'description': [
                'Touch Control Headphone with 40 Hours Playtime, Comfort Fit, Latest Bluetooth v5.3',
                'Provides a massive battery backup of upto 15 hours for a superior playback time with 40mm dynamic drivers',
                'The stylish BassHeads 100 superior coated wired earphones with powerful 10mm dynamic driver',
                'With up to 50-hour battery life and quick charging, great sound quality customizable with EQ Custom',
                'Comfortable Design with 60hrs Playback Time, Superior Sound Quality, and Multi Connectivity Options',
                '40mm Drivers, Lightweight Build, Remote Control, Unidirectional Mic, and Foldable Design',
                '70H Playtime, 40mm Bass Drivers, Zen ENC Mic, Type-C Fast Charging, 4 EQ Modes, Bluetooth 5.4'
            ],
            'availability': [33, 53, 58, 53, 74, 41, 27],
            'loyaltypoints': [229, 149, 29, 398, 69, 89, 179],
            'type': ['Over-Ear', 'On-Ear', 'In-Ear', 'On-Ear', 'Over-Ear', 'Over-Ear', 'Over-Ear'],
            'connectivity': ['Wireless', 'Wireless', 'Wired', 'Wireless', 'Wireless', 'Wired', 'Wireless'],
            'battery_life': [40, 15, 0, 50, 60, 0, 70],
        }
        
        df = pd.DataFrame(data)
        
        # Extract base model name
        df['base_model'] = df['name'].apply(lambda x: x.split(',')[0].strip())
        
        # Combine text features into a single column
        df['combined_text'] = df['name'] + ' ' + df['brand'] + ' ' + df['description'] + ' ' + df['category']
        
        # Normalize numerical features
        scaler = MinMaxScaler()
        df[['price_normalized', 'rating_normalized', 'reviews_normalized']] = scaler.fit_transform(df[['price', 'rating', 'reviews']])
        
        # TF-IDF for text features
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        
        # Combine TF-IDF and numerical features
        numerical_features = df[['price_normalized', 'rating_normalized', 'reviews_normalized']].values
        feature_matrix = hstack([tfidf_matrix, numerical_features])
        
        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
        
        return df, cosine_sim

df, cosine_sim = load_and_process_data()

# Function to get recommendations
def get_recommendations(product_name, top_n=5, price_range=None, min_rating=0, connectivity=None, headphone_type=None, brand=None):
    """
    Get top N recommendations for a product with filtering options.
    """
    try:
        if product_name not in df['name'].values:
            return []
        
        product_index = df[df['name'] == product_name].index[0]
        base_model = df.loc[product_index, 'base_model']
        sim_scores = list(enumerate(cosine_sim[product_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        filtered_recommendations = []
        seen_models = {base_model}
        
        for i, score in sim_scores:
            # Skip the product itself
            if i == product_index:
                continue
                
            current_product = df.iloc[i]
            current_base_model = current_product['base_model']
            
            # Apply filters
            if current_base_model in seen_models:
                continue
                
            if price_range and (current_product['price'] < price_range[0] or current_product['price'] > price_range[1]):
                continue
                
            if current_product['rating'] < min_rating:
                continue
                
            if connectivity and current_product['connectivity'] != connectivity:
                continue
                
            if headphone_type and current_product['type'] != headphone_type:
                continue
                
            if brand and current_product['brand'] != brand:
                continue
            
            # Add to recommendations
            filtered_recommendations.append({
                'index': i,
                'name': current_product['name'],
                'brand': current_product['brand'],
                'price': current_product['price'],
                'rating': current_product['rating'],
                'reviews': current_product['reviews'],
                'image_url': current_product['image_url'],
                'description': current_product['description'],
                'type': current_product['type'],
                'connectivity': current_product['connectivity'],
                'battery_life': current_product['battery_life'],
                'availability': current_product['availability'],
                'loyaltypoints': current_product['loyaltypoints'],
                'similarity': score[1]
            })
            
            seen_models.add(current_base_model)
            
            if len(filtered_recommendations) >= top_n:
                break
        
        return filtered_recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return []

# Function to search products
def search_products(query, df):
    if not query:
        return df
    
    query = query.lower()
    mask = (
        df['name'].str.lower().str.contains(query) | 
        df['brand'].str.lower().str.contains(query) | 
        df['description'].str.lower().str.contains(query)
    )
    return df[mask]

# App header with logo and dark mode toggle
col1, col2, col3 = st.columns([1, 5, 1])
with col1:
    st.image("https://img.icons8.com/color/96/000000/headphones.png", width=80)
with col2:
    st.markdown('<p class="main-header">SoundMatch</p>', unsafe_allow_html=True)
    st.markdown('<p>Find your perfect headphone match with AI-powered recommendations</p>', unsafe_allow_html=True)
with col3:
    dark_mode_label = "🌙 Dark" if not st.session_state.dark_mode else "☀️ Light"
    if st.button(dark_mode_label):
        toggle_dark_mode()
        st.experimental_rerun()

# Create navigation menu
selected = option_menu(
    menu_title=None,
    options=["Discover", "Compare", "Explore", "Insights", "About"],
    icons=["search", "bar-chart", "grid", "graph-up", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "var(--primary-color)", "font-size": "14px"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px",
            "--hover-color": "rgba(124, 58, 237, 0.1)",
        },
        "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"},
    }
)

# Product Detail Modal
if st.session_state.show_product_detail and st.session_state.selected_product:
    try:
        product = df[df['name'] == st.session_state.selected_product].iloc[0]
        
        with st.container():
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Product Details</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(product['image_url'], width=300)
                
                # Availability status
                availability_status, availability_class = get_availability_status(product['availability'])
                st.markdown(f"""
                <div class="availability-indicator {availability_class}">
                    <span>Availability: {availability_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Loyalty points
                st.markdown(f"""
                <div class="loyalty-points">
                    <span>🏆 Loyalty Points: {product['loyaltypoints']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Close button
                if st.button("← Back to Products", key="close_detail"):
                    close_product_detail()
                    st.experimental_rerun()
            
            with col2:
                st.markdown(f'<h2 class="product-detail-title">{product["name"]}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p class="product-detail-brand">By {product["brand"]}</p>', unsafe_allow_html=True)
                
                # Rating and price
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="price-tag" style="font-size: 1.2rem;">₹{product["price"]}</div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown(render_star_rating(product['rating']), unsafe_allow_html=True)
                    st.markdown(f'<span>({product["reviews"]:,} reviews)</span>', unsafe_allow_html=True)
                
                # Type and connectivity badges
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <span class="badge badge-primary">{product['type']}</span>
                    <span class="badge badge-secondary">{product['connectivity']}</span>
                    {f'<span class="badge badge-accent">{product["battery_life"]}h Battery</span>' if product['battery_life'] > 0 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Description
                st.markdown('<h3>Description</h3>', unsafe_allow_html=True)
                st.markdown(f'<p class="product-detail-description">{product["description"]}</p>', unsafe_allow_html=True)
                
                # Features
                features = extract_features(product['description'])
                if features:
                    st.markdown('<h3>Key Features</h3>', unsafe_allow_html=True)
                    feature_html = '<ul class="feature-list">'
                    for feature in features:
                        feature_html += f'<li class="feature-item"><span class="feature-icon">✓</span> {feature}</li>'
                    feature_html += '</ul>'
                    st.markdown(feature_html, unsafe_allow_html=True)
            
            # Similar products
            st.markdown('<h3>Similar Products You Might Like</h3>', unsafe_allow_html=True)
            recommendations = get_recommendations(product['name'], top_n=3)
            
            if recommendations:
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="product-card animate-fade-in" style="animation-delay: {i * 0.2}s;">
                            <div class="card-overlay"></div>
                            <div class="product-image-container">
                                <img src="{rec['image_url']}" class="product-image" alt="{rec['name']}">
                            </div>
                            <h3 class="product-title">{rec['name'][:50]}...</h3>
                            <p class="product-brand">{rec['brand']}</p>
                            <div class="product-meta">
                                <span class="price-tag">₹{rec['price']}</span>
                                <span class="rating-tag">★ {rec['rating']}</span>
                            </div>
                            <button class="custom-button" style="width: 100%;" onclick="alert('View product details')">
                                View Details
                            </button>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No similar products found.")
    
    except Exception as e:
        st.error(f"Error displaying product details: {e}")
        close_product_detail()

# Discover tab
elif selected == "Discover":
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<p class="sub-header">Find Your Perfect Match</p>', unsafe_allow_html=True)
        
        # Filters in a stylable container
        with stylable_container(
            key="filter_container",
            css_styles="""
                {
                    background-color: var(--card-bg-color);
                    border-radius: 16px;
                    padding: 1.5rem;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
                }
            """
        ):
            st.markdown("### 🎛️ Filters")
            
            # Product selection
            product_name = st.selectbox(
                "Select a product:",
                df['name'].tolist(),
                index=0
            )
            
            # Number of recommendations
            top_n = st.slider("Number of recommendations", min_value=1, max_value=6, value=3)
            
            # Price range filter
            st.markdown("#### Price Range")
            min_price = int(df['price'].min())
            max_price = int(df['price'].max())
            price_range = st.slider(
                "Select price range (₹)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
            
            # Rating filter
            min_rating = st.slider("Minimum rating", min_value=3.0, max_value=5.0, value=3.5, step=0.1)
            
            # Brand filter
            brand_options = ["Any"] + sorted(df['brand'].unique().tolist())
            brand = st.selectbox("Brand", brand_options)
            brand_filter = None if brand == "Any" else brand
            
            # Connectivity filter
            connectivity = st.radio(
                "Connectivity type",
                options=["Any", "Wireless", "Wired"],
                horizontal=True
            )
            connectivity_filter = None if connectivity == "Any" else connectivity
            
            # Headphone type filter
            headphone_type = st.radio(
                "Headphone type",
                options=["Any", "Over-Ear", "On-Ear", "In-Ear"],
                horizontal=True
            )
            type_filter = None if headphone_type == "Any" else headphone_type
            
            # Find button
            find_button = st.button(
                "Find Similar Products",
                type="primary",
                use_container_width=True
            )
        
        # Display Lottie animation
        streamlit_lottie(headphone_animation, height=200, key="headphone_animation")
        
        # Quick tips
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - Select a product you already like as a starting point
            - Adjust price range to find alternatives in your budget
            - Use the connectivity filter to find specific types of headphones
            - Higher similarity scores indicate closer matches to your selected product
            """)
    
    with col2:
        if find_button:
            with st.spinner("Finding the perfect matches for you..."):
                # Simulate processing time for effect
                time.sleep(1)
                
                # Get recommendations
                recommendations = get_recommendations(
                    product_name,
                    top_n=top_n,
                    price_range=price_range,
                    min_rating=min_rating,
                    connectivity=connectivity_filter,
                    headphone_type=type_filter,
                    brand=brand_filter
                )
            
            if recommendations:
                # Display selected product
                st.markdown('<p class="sub-header">Selected Product</p>', unsafe_allow_html=True)
                selected_product = df[df['name'] == product_name].iloc[0]
                
                with stylable_container(
                    key="selected_product",
                    css_styles="""
                        {
                            background-color: var(--card-bg-color);
                            border-radius: 16px;
                            padding: 1.5rem;
                            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
                            border-left: 5px solid var(--primary-color);
                            margin-bottom: 2rem;
                        }
                    """
                ):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(selected_product['image_url'], width=200)
                    with col2:
                        st.markdown(f"### {selected_product['name']}")
                        st.markdown(f"**Brand**: {selected_product['brand']}")
                        
                        # Display metrics in a row
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Price", f"₹{selected_product['price']}")
                        with metrics_col2:
                            st.metric("Rating", f"{selected_product['rating']} ★")
                        with metrics_col3:
                            st.metric("Reviews", f"{selected_product['reviews']:,}")
                        
                        style_metric_cards()
                        
                        # Display badges
                        st.markdown(f"""
                        <span class="badge badge-primary">{selected_product['type']}</span>
                        <span class="badge badge-secondary">{selected_product['connectivity']}</span>
                        """, unsafe_allow_html=True)
                        
                        if selected_product['battery_life'] > 0:
                            st.markdown(f"""
                            <span class="badge badge-accent">{selected_product['battery_life']}h Battery</span>
                            """, unsafe_allow_html=True)
                        
                        # View details button
                        if st.button("View Details", key="view_selected_details"):
                            show_product_detail(selected_product['name'])
                            st.experimental_rerun()
                
                # Display recommendations
                st.markdown('<p class="sub-header">Recommended Products</p>', unsafe_allow_html=True)
                
                # Create a grid for recommendations
                cols = st.columns(3)
                for i, rec in enumerate(recommendations):
                    with cols[i % 3]:
                        # Add animation delay based on index
                        animation_delay = i * 0.2
                        
                        st.markdown(f"""
                        <div class="product-card animate-fade-in" style="animation-delay: {animation_delay}s;">
                            <div class="card-overlay"></div>
                            <div class="product-image-container">
                                <img src="{rec['image_url']}" class="product-image" alt="{rec['name']}">
                            </div>
                            <h3 class="product-title">{rec['name'][:50]}...</h3>
                            <p class="product-brand">{rec['brand']}</p>
                            <div class="product-meta">
                                <span class="price-tag">₹{rec['price']}</span>
                                <span class="rating-tag">★ {rec['rating']}</span>
                            </div>
                            <p class="product-description">{rec['description'][:100]}...</p>
                            <div class="progress-container">
                                <div class="progress-bar" style="width: {int(rec['similarity'] * 100)}%;"></div>
                            </div>
                            <p style="text-align: right;"><b>Match:</b> {int(rec['similarity'] * 100)}%</p>
                            <div class="product-card-footer">
                                <button class="custom-button" style="width: 100%;" id="view_details_{i}">
                                    View Details
                                </button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add a button with the same ID to handle the click
                        if st.button(f"View Details", key=f"view_details_btn_{i}", use_container_width=True):
                            show_product_detail(rec['name'])
                            st.experimental_rerun()
                
                # Visualization of recommendations
                st.markdown('<p class="sub-header">Recommendation Analysis</p>', unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Price Comparison", "Match Score", "Feature Radar"])
                
                with viz_tab1:
                    # Price comparison chart
                    price_data = [{'Product': rec['name'][:20] + "...", 'Price': rec['price']} for rec in recommendations]
                    price_data.append({'Product': selected_product['name'][:20] + "...", 'Price': selected_product['price']})
                    price_df = pd.DataFrame(price_data)
                    
                    fig = px.bar(
                        price_df, 
                        x='Product', 
                        y='Price', 
                        title='Price Comparison',
                        color='Product',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab2:
                    # Similarity score chart
                    sim_data = [{'Product': rec['name'][:20] + "...", 'Match Score': rec['similarity'] * 100} for rec in recommendations]
                    sim_df = pd.DataFrame(sim_data)
                    
                    fig = px.bar(
                        sim_df, 
                        x='Product', 
                        y='Match Score', 
                        title='Similarity Match Score (%)',
                        color='Match Score',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab3:
                    # Radar chart for feature comparison
                    fig = go.Figure()
                    
                    # Add selected product
                    selected_values = [
                        selected_product['price_normalized'] * 100,
                        selected_product['rating'] * 20,
                        selected_product['battery_life'] / 70 * 100 if selected_product['battery_life'] > 0 else 0,
                        selected_product['availability'],
                        selected_product['loyaltypoints'] / 400 * 100
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=selected_values,
                        theta=['Price', 'Rating', 'Battery Life', 'Availability', 'Loyalty Points'],
                        fill='toself',
                        name=selected_product['name'][:20] + "...",
                        line=dict(color='rgba(124, 58, 237, 0.8)'),
                        fillcolor='rgba(124, 58, 237, 0.2)'
                    ))
                    
                    # Add recommended products
                    colors = ['rgba(236, 72, 153, 0.8)', 'rgba(79, 70, 229, 0.8)', 'rgba(16, 185, 129, 0.8)']
                    fill_colors = ['rgba(236, 72, 153, 0.2)', 'rgba(79, 70, 229, 0.2)', 'rgba(16, 185, 129, 0.2)']
                    
                    for i, rec in enumerate(recommendations[:3]):  # Limit to 3 for clarity
                        rec_values = [
                            rec['price'] / max_price * 100,
                            rec['rating'] * 20,
                            rec['battery_life'] / 70 * 100 if rec['battery_life'] > 0 else 0,
                            rec['availability'],
                            rec['loyaltypoints'] / 400 * 100
                        ]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=rec_values,
                            theta=['Price', 'Rating', 'Battery Life', 'Availability', 'Loyalty Points'],
                            fill='toself',
                            name=rec['name'][:20] + "...",
                            line=dict(color=colors[i % len(colors)]),
                            fillcolor=fill_colors[i % len(fill_colors)]
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        showlegend=True,
                        title="Feature Comparison",
                        template="plotly_white",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Display error message if no recommendations found
                st.markdown("""
                <div class="error-container animate-fade-in">
                    <div class="error-icon">😕</div>
                    <h2 class="error-title">No Recommendations Found</h2>
                    <p class="error-message">We couldn't find any products that match your filters. Try adjusting your criteria to see more results.</p>
                    <button class="custom-button">Reset Filters</button>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Display welcome message and instructions
            st.markdown("""
            <div style="text-align: center; padding: 50px 20px; background-color: var(--card-bg-color); border-radius: 16px; margin-top: 50px;" class="animate-fade-in">
                <img src="https://img.icons8.com/color/96/000000/headphones.png" width="100">
                <h2 style="margin-top: 20px;">Welcome to SoundMatch!</h2>
                <p style="margin: 20px 0; font-size: 1.2rem;">
                    Find your perfect headphone match with our AI-powered recommendation system.
                </p>
                <p>Select a product and click "Find Similar Products" to get started.</p>
            </div>
            """, unsafe_allow_html=True)

# Compare tab
elif selected == "Compare":
    st.markdown('<p class="sub-header">Compare Headphones</p>', unsafe_allow_html=True)
    
    # Select products to compare
    col1, col2 = st.columns(2)
    with col1:
        product1 = st.selectbox("Select first product", df['name'].tolist(), index=0, key="product1")
    with col2:
        product2 = st.selectbox("Select second product", df['name'].tolist(), index=1 if len(df) > 1 else 0, key="product2")
    
    # Add a third product option
    add_third = st.checkbox("Add a third product to compare")
    if add_third:
        product3 = st.selectbox("Select third product", df['name'].tolist(), index=2 if len(df) > 2 else 0, key="product3")
    
    # Get product data
    product1_data = df[df['name'] == product1].iloc[0]
    product2_data = df[df['name'] == product2].iloc[0]
    if add_third:
        product3_data = df[df['name'] == product3].iloc[0]
    
    # Display comparison
    if st.button("Compare Products", type="primary"):
        with st.spinner("Generating comparison..."):
            time.sleep(1)  # Simulate processing
            
            # Display product cards side by side
            if add_third:
                cols = st.columns(3)
            else:
                cols = st.columns(2)
            
            with cols[0]:
                st.markdown(f"""
                <div class="product-card animate-fade-in">
                    <div class="card-overlay"></div>
                    <div class="product-image-container">
                        <img src="{product1_data['image_url']}" class="product-image" alt="{product1_data['name']}">
                    </div>
                    <h3 class="product-title">{product1_data['name'][:50]}...</h3>
                    <p class="product-brand">{product1_data['brand']}</p>
                    <div class="product-meta">
                        <span class="price-tag">₹{product1_data['price']}</span>
                        <span class="rating-tag">★ {product1_data['rating']}</span>
                    </div>
                    <p><b>Type:</b> {product1_data['type']}</p>
                    <p><b>Connectivity:</b> {product1_data['connectivity']}</p>
                    <p><b>Battery Life:</b> {product1_data['battery_life']} hours</p>
                    <p><b>Features:</b> {product1_data['description'][:100]}...</p>
                    <div class="product-card-footer">
                        <button class="custom-button" style="width: 100%;" id="view_details_p1">
                            View Details
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("View Details", key="view_details_p1_btn"):
                    show_product_detail(product1_data['name'])
                    st.experimental_rerun()

            with cols[1]:
                st.markdown(f"""
                <div class="product-card animate-fade-in" style="animation-delay: 0.2s;">
                    <div class="card-overlay"></div>
                    <div class="product-image-container">
                        <img src="{product2_data['image_url']}" class="product-image" alt="{product2_data['name']}">
                    </div>
                    <h3 class="product-title">{product2_data['name'][:50]}...</h3>
                    <p class="product-brand">{product2_data['brand']}</p>
                    <div class="product-meta">
                        <span class="price-tag">₹{product2_data['price']}</span>
                        <span class="rating-tag">★ {product2_data['rating']}</span>
                    </div>
                    <p><b>Type:</b> {product2_data['type']}</p>
                    <p><b>Connectivity:</b> {product2_data['connectivity']}</p>
                    <p><b>Battery Life:</b> {product2_data['battery_life']} hours</p>
                    <p><b>Features:</b> {product2_data['description'][:100]}...</p>
                    <div class="product-card-footer">
                        <button class="custom-button" style="width: 100%;" id="view_details_p2">
                            View Details
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("View Details", key="view_details_p2_btn"):
                    show_product_detail(product2_data['name'])
                    st.experimental_rerun()

            if add_third:
                with cols[2]:
                    st.markdown(f"""
                    <div class="product-card animate-fade-in" style="animation-delay: 0.4s;">
                        <div class="card-overlay"></div>
                        <div class="product-image-container">
                            <img src="{product3_data['image_url']}" class="product-image" alt="{product3_data['name']}">
                        </div>
                        <h3 class="product-title">{product3_data['name'][:50]}...</h3>
                        <p class="product-brand">{product3_data['brand']}</p>
                        <div class="product-meta">
                            <span class="price-tag">₹{product3_data['price']}</span>
                            <span class="rating-tag">★ {product3_data['rating']}</span>
                        </div>
                        <p><b>Type:</b> {product3_data['type']}</p>
                        <p><b>Connectivity:</b> {product3_data['connectivity']}</p>
                        <p><b>Battery Life:</b> {product3_data['battery_life']} hours</p>
                        <p><b>Features:</b> {product3_data['description'][:100]}...</p>
                        <div class="product-card-footer">
                            <button class="custom-button" style="width: 100%;" id="view_details_p3">
                                View Details
                            </button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("View Details", key="view_details_p3_btn"):
                        show_product_detail(product3_data['name'])
                        st.experimental_rerun()

            # Feature comparison table
            st.markdown('<p class="sub-header">Feature Comparison</p>', unsafe_allow_html=True)

            # Create comparison data
            comparison_data = {
                'Feature': ['Price', 'Rating', 'Reviews', 'Type', 'Connectivity', 'Battery Life', 'Availability', 'Loyalty Points'],
                product1_data['name'][:20] + "...": [
                    f"₹{product1_data['price']}",
                    f"{product1_data['rating']} ★",
                    f"{product1_data['reviews']:,}",
                    product1_data['type'],
                    product1_data['connectivity'],
                    f"{product1_data['battery_life']} hours" if product1_data['battery_life'] > 0 else "N/A",
                    f"{product1_data['availability']}%",
                    product1_data['loyaltypoints']
                ],
                product2_data['name'][:20] + "...": [
                    f"₹{product2_data['price']}",
                    f"{product2_data['rating']} ★",
                    f"{product2_data['reviews']:,}",
                    product2_data['type'],
                    product2_data['connectivity'],
                    f"{product2_data['battery_life']} hours" if product2_data['battery_life'] > 0 else "N/A",
                    f"{product2_data['availability']}%",
                    product2_data['loyaltypoints']
                ]
            }

            if add_third:
                comparison_data[product3_data['name'][:20] + "..."] = [
                    f"₹{product3_data['price']}",
                    f"{product3_data['rating']} ★",
                    f"{product3_data['reviews']:,}",
                    product3_data['type'],
                    product3_data['connectivity'],
                    f"{product3_data['battery_life']} hours" if product3_data['battery_life'] > 0 else "N/A",
                    f"{product3_data['availability']}%",
                    product3_data['loyaltypoints']
                ]

            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)

            # Visual comparisons
            st.markdown('<p class="sub-header">Visual Comparison</p>', unsafe_allow_html=True)

            # Create tabs for different visualizations
            viz_tab1, viz_tab2 = st.tabs(["Price & Rating", "Feature Radar"])

            with viz_tab1:
                # Create a subplot with 2 y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Add price bars
                products = [product1_data['name'][:20] + "...", product2_data['name'][:20] + "..."]
                prices = [product1_data['price'], product2_data['price']]
                ratings = [product1_data['rating'], product2_data['rating']]

                if add_third:
                    products.append(product3_data['name'][:20] + "...")
                    prices.append(product3_data['price'])
                    ratings.append(product3_data['rating'])

                # Add price bars
                fig.add_trace(
                    go.Bar(
                        x=products,
                        y=prices,
                        name="Price (₹)",
                        marker_color='rgba(124, 58, 237, 0.7)',
                        text=prices,
                        textposition='auto',
                    ),
                    secondary_y=False,
                )

                # Add rating line
                fig.add_trace(
                    go.Scatter(
                        x=products,
                        y=ratings,
                        name="Rating",
                        marker=dict(size=12),
                        line=dict(width=4, color='rgba(236, 72, 153, 0.7)'),
                        mode='lines+markers+text',
                        text=ratings,
                        textposition='top center',
                    ),
                    secondary_y=True,
                )

                # Set titles
                fig.update_layout(
                    title_text="Price vs Rating Comparison",
                    template="plotly_white",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # Set y-axes titles
                fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
                fig.update_yaxes(title_text="Rating", secondary_y=True, range=[3, 5])

                st.plotly_chart(fig, use_container_width=True)

            with viz_tab2:
                # Radar chart for feature comparison
                fig = go.Figure()

                # Prepare data for radar chart
                products_data = [product1_data, product2_data]
                if add_third:
                    products_data.append(product3_data)

                colors = ['rgba(124, 58, 237, 0.8)', 'rgba(236, 72, 153, 0.8)', 'rgba(79, 70, 229, 0.8)']
                fill_colors = ['rgba(124, 58, 237, 0.2)', 'rgba(236, 72, 153, 0.2)', 'rgba(79, 70, 229, 0.2)']

                for i, product in enumerate(products_data):
                    # Normalize values for radar chart
                    price_norm = 100 - (product['price'] / max(df['price']) * 100)  # Invert so lower price is better
                    rating_norm = product['rating'] / 5 * 100
                    reviews_norm = min(product['reviews'] / max(df['reviews']) * 100, 100)
                    battery_norm = product['battery_life'] / max(df['battery_life']) * 100 if product['battery_life'] > 0 else 0
                    avail_norm = product['availability']
                    loyalty_norm = product['loyaltypoints'] / max(df['loyaltypoints']) * 100

                    values = [price_norm, rating_norm, reviews_norm, battery_norm, avail_norm, loyalty_norm]

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Price', 'Rating', 'Reviews', 'Battery Life', 'Availability', 'Loyalty Points'],
                        fill='toself',
                        name=product['name'][:20] + "...",
                        line=dict(color=colors[i]),
                        fillcolor=fill_colors[i]
                    ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title="Feature Comparison (Higher is Better)",
                    template="plotly_white",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)
    else:
        # Display animation when no comparison is active
        col1, col2 = st.columns([1, 1])
        with col1:
            streamlit_lottie(compare_animation, height=300, key="compare_animation")
        with col2:
            st.markdown("""
            <div style="padding: 30px 20px;">
                <h2>Compare Headphones Side by Side</h2>
                <p style="margin: 20px 0; font-size: 1.1rem;">
                    Select two or three products to compare their features, specifications, and value.
                </p>
                <ul>
                    <li>Compare prices and ratings</li>
                    <li>Analyze feature differences</li>
                    <li>Visualize performance metrics</li>
                    <li>Make informed purchasing decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Explore tab
elif selected == "Explore":
    st.markdown('<p class="sub-header">Explore All Headphones</p>', unsafe_allow_html=True)

    # Search bar
    st.markdown("""
    <div class="search-container animate-fade-in">
        <span class="search-icon">🔍</span>
        <input type="text" class="search-input" placeholder="Search for headphones by name, brand, or features..." id="search_input">
    </div>
    """, unsafe_allow_html=True)

    # Create a Streamlit text input that will update when the user types
    search_query = st.text_input("", value=st.session_state.search_query, label_visibility="collapsed", key="search_box")

    # Update session state
    st.session_state.search_query = search_query

    # Filters in a horizontal layout
    with stylable_container(
        key="filter_panel",
        css_styles="""
            {
                background-color: var(--card-bg-color);
                border-radius: 16px;
                padding: 1.5rem;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
                margin-bottom: 1.5rem;
            }
        """
    ):
        st.markdown('<p class="filter-title">Filter & Sort</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Price (Low to High)", "Price (High to Low)", "Rating (High to Low)", "Reviews (High to Low)"]
            )
        with col2:
            brand_filter = st.multiselect("Filter by brand:", df['brand'].unique())
        with col3:
            type_filter = st.multiselect("Filter by type:", df['type'].unique())
        with col4:
            connectivity_filter = st.multiselect("Filter by connectivity:", df['connectivity'].unique())

        # Price range slider
        price_range = st.slider(
            "Price range (₹):",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].min()), int(df['price'].max()))
        )

        # Filter buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Apply Filters", type="primary", use_container_width=True):
                st.session_state.filter_applied = True
        with col2:
            if st.button("Reset Filters", use_container_width=True):
                reset_filters()
                st.experimental_rerun()

    # Apply filters
    filtered_df = df.copy()

    # Search filter
    if search_query:
        filtered_df = search_products(search_query, filtered_df)

    # Price filter
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

    # Brand filter
    if brand_filter:
        filtered_df = filtered_df[filtered_df['brand'].isin(brand_filter)]

    # Type filter
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]

    # Connectivity filter
    if connectivity_filter:
        filtered_df = filtered_df[filtered_df['connectivity'].isin(connectivity_filter)]

    # Apply sorting
    if sort_by == "Price (Low to High)":
        filtered_df = filtered_df.sort_values('price')
    elif sort_by == "Price (High to Low)":
        filtered_df = filtered_df.sort_values('price', ascending=False)
    elif sort_by == "Rating (High to Low)":
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    else:  # Reviews (High to Low)
        filtered_df = filtered_df.sort_values('reviews', ascending=False)

    # Display product count
    st.markdown(f"### Showing {len(filtered_df)} products")

    # Display products in a grid
    if not filtered_df.empty:
        # Create a grid layout
        st.markdown('<div class="product-grid">', unsafe_allow_html=True)

        # Display each product
        for i, (_, product) in enumerate(filtered_df.iterrows()):
            # Add animation delay based on index
            animation_delay = (i % 9) * 0.1  # Reset delay every 9 items for better performance

            st.markdown(f"""
            <div class="product-card animate-fade-in" style="animation-delay: {animation_delay}s;">
                <div class="card-overlay"></div>
                <div class="product-image-container">
                    <img src="{product['image_url']}" class="product-image" alt="{product['name']}">
                </div>
                <h3 class="product-title">{product['name'][:50]}...</h3>
                <p class="product-brand">{product['brand']}</p>
                <div class="product-meta">
                    <span class="price-tag">₹{product['price']}</span>
                    <span class="rating-tag">★ {product['rating']}</span>
                </div>
                <div style="margin: 0.5rem 0;">
                    <span class="badge badge-primary">{product['type']}</span>
                    <span class="badge badge-secondary">{product['connectivity']}</span>
                    {f'<span class="badge badge-accent">{product["battery_life"]}h Battery</span>' if product['battery_life'] > 0 else ''}
                </div>
                <p class="product-description">{product['description'][:100]}...</p>
                <div class="product-card-footer">
                    <button class="custom-button" style="width: 100%;" id="view_details_explore_{i}">
                        View Details
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Add a hidden button with the same ID to handle the click
            if st.button(f"View Details", key=f"view_details_explore_btn_{i}", use_container_width=True, label_visibility="collapsed"):
                show_product_detail(product['name'])
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display empty state
        st.markdown("""
        <div class="empty-state animate-fade-in">
            <div class="empty-state-icon">🔍</div>
            <h2 class="empty-state-title">No Products Found</h2>
            <p class="empty-state-message">We couldn't find any products that match your search or filters. Try adjusting your criteria to see more results.</p>
            <button class="custom-button">Reset Filters</button>
        </div>
        """, unsafe_allow_html=True)

# Insights tab
elif selected == "Insights":
    st.markdown('<p class="sub-header">Market Insights</p>', unsafe_allow_html=True)

    # Create tabs for different insights
    insight_tab1, insight_tab2, insight_tab3 = st.tabs(["Price Analysis", "Brand Comparison", "Feature Trends"])

    with insight_tab1:
        st.markdown("### Price Distribution Analysis")

        # Price histogram
        fig = px.histogram(
            df,
            x="price",
            nbins=10,
            color_discrete_sequence=['rgba(124, 58, 237, 0.7)'],
            title="Price Distribution of Headphones",
            labels={"price": "Price (₹)", "count": "Number of Products"}
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Price vs Rating scatter plot
        fig = px.scatter(
            df,
            x="price",
            y="rating",
            size="reviews",
            color="brand",
            hover_name="name",
            title="Price vs Rating Relationship",
            labels={"price": "Price (₹)", "rating": "Rating", "reviews": "Number of Reviews"}
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Price insights
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Price", f"₹{df['price'].mean():.2f}")
            st.metric("Lowest Price", f"₹{df['price'].min()}")
        with col2:
            st.metric("Median Price", f"₹{df['price'].median()}")
            st.metric("Highest Price", f"₹{df['price'].max()}")

        style_metric_cards()

    with insight_tab2:
        st.markdown("### Brand Comparison")

        # Brand market share
        brand_counts = df['brand'].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']

        fig = px.pie(
            brand_counts,
            values='Count',
            names='Brand',
            title="Brand Market Share",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Average rating by brand
        brand_ratings = df.groupby('brand')['rating'].mean().reset_index()
        brand_ratings = brand_ratings.sort_values('rating', ascending=False)

        fig = px.bar(
            brand_ratings,
            x='brand',
            y='rating',
            title="Average Rating by Brand",
            labels={"brand": "Brand", "rating": "Average Rating"},
            color='rating',
            color_continuous_scale=px.colors.sequential.Viridis,
            text_auto='.1f'
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Brand comparison table
        brand_stats = df.groupby('brand').agg({
            'price': ['mean', 'min', 'max'],
            'rating': 'mean',
            'reviews': 'sum'
        }).reset_index()

        brand_stats.columns = ['Brand', 'Avg Price', 'Min Price', 'Max Price', 'Avg Rating', 'Total Reviews']
        brand_stats['Avg Price'] = brand_stats['Avg Price'].round(2)
        brand_stats['Avg Rating'] = brand_stats['Avg Rating'].round(2)

        st.dataframe(brand_stats, use_container_width=True)

    with insight_tab3:
        st.markdown("### Feature Trends")

        # Connectivity type distribution
        conn_counts = df['connectivity'].value_counts().reset_index()
        conn_counts.columns = ['Connectivity', 'Count']

        fig = px.pie(
            conn_counts,
            values='Count',
            names='Connectivity',
            title="Connectivity Type Distribution",
            color_discrete_sequence=['rgba(124, 58, 237, 0.7)', 'rgba(236, 72, 153, 0.7)']
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=350
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig, use_container_width=True)

        # Headphone type distribution
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']

        fig = px.pie(
            type_counts,
            values='Count',
            names='Type',
            title="Headphone Type Distribution",
            color_discrete_sequence=['rgba(79, 70, 229, 0.7)', 'rgba(16, 185, 129, 0.7)', 'rgba(245, 158, 11, 0.7)']
        )

        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            height=350
        )

        with col2:
            st.plotly_chart(fig, use_container_width=True)

        # Battery life comparison for wireless headphones
        wireless_df = df[df['connectivity'] == 'Wireless'].copy()

        if not wireless_df.empty:
            fig = px.bar(
                wireless_df,
                x='name',
                y='battery_life',
                color='brand',
                title="Battery Life Comparison (Wireless Headphones)",
                labels={"name": "Product", "battery_life": "Battery Life (hours)", "brand": "Brand"},
                text_auto=True
            )

            fig.update_layout(
                template="plotly_white",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                height=450,
                xaxis={'categoryorder':'total descending'}
            )

            fig.update_xaxes(tickangle=45, title_text="")

            st.plotly_chart(fig, use_container_width=True)

        # Feature trends insights
        st.markdown("### Key Feature Insights")

        col1, col2, col3 = st.columns(3)
        with col1:
            wireless_percent = len(df[df['connectivity'] == 'Wireless']) / len(df) * 100
            st.metric("Wireless Headphones", f"{wireless_percent:.1f}%")
        with col2:
            avg_battery = df[df['battery_life'] > 0]['battery_life'].mean()
            st.metric("Avg Battery Life", f"{avg_battery:.1f} hours")
        with col3:
            over_ear_percent = len(df[df['type'] == 'Over-Ear']) / len(df) * 100
            st.metric("Over-Ear Headphones", f"{over_ear_percent:.1f}%")

        style_metric_cards()

# About tab
elif selected == "About":
    st.markdown('<p class="sub-header">About SoundMatch</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://img.icons8.com/color/96/000000/headphones.png", width=200)

    with col2:
        st.markdown("""
        ### How SoundMatch Works

        SoundMatch is an AI-powered headphone recommendation system that helps you find the perfect headphones based on your preferences and needs.

        #### The recommendation engine uses:

        1. **Content-based filtering**: Analyzes product features, descriptions, and specifications
        2. **Similarity metrics**: Uses advanced algorithms to find products similar to ones you like
        3. **Multi-factor analysis**: Considers price, ratings, features, and more for balanced recommendations

        #### Features of this app:

        - **Smart recommendations**: Find headphones similar to ones you already like
        - **Detailed comparisons**: Compare products side by side to make informed decisions
        - **Market insights**: Explore trends and patterns in the headphone market
        - **Interactive visualizations**: Visualize data to better understand product differences
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # How to use section
    st.markdown("### How to Use SoundMatch")

    st.markdown("""
    1. **Discover**: Find headphones similar to ones you like
       - Select a product you're interested in
       - Apply filters to narrow down recommendations
       - Explore the recommended products with detailed information

    2. **Compare**: Compare headphones side by side
       - Select two or three products to compare
       - View detailed feature comparisons
       - Analyze visual representations of differences

    3. **Explore**: Browse all available headphones
       - Sort and filter the product catalog
       - View detailed product information
       - Find products that match your specific needs

    4. **Insights**: Explore market trends and patterns
       - Analyze price distributions
       - Compare brands and their performance
       - Discover feature trends in the market
    """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Tips section
    st.markdown("### Tips for Finding Your Perfect Headphones")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Consider Your Use Case

        - **Commuting/Travel**: Look for wireless, noise-cancelling options with good battery life
        - **Office/Work**: Consider comfort for long wear and good microphone quality
        - **Sports/Fitness**: Look for water resistance, secure fit, and durability
        - **Home/Entertainment**: Focus on sound quality and comfort over portability
        """)

    with col2:
        st.markdown("""
        #### Key Features to Consider

        - **Sound Quality**: Higher price often (but not always) means better sound
        - **Comfort**: Over-ear for comfort, in-ear for portability
        - **Battery Life**: For wireless options, look for at least 20+ hours
        - **Connectivity**: Consider Bluetooth version for wireless options
        - **Additional Features**: Noise cancellation, water resistance, foldable design
        """)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: var(--light-text); padding: 20px 0;">
    <p>© 2023 SoundMatch | Headphone Recommendation System | Built with Streamlit</p>
    <p style="font-size: 0.8rem;">Data sourced from various e-commerce platforms. All product information is for demonstration purposes only.</p>
</div>
""", unsafe_allow_html=True)

# Add JavaScript for animations and interactions
st.markdown("""
<script>
    // Animate progress bars
    document.addEventListener('DOMContentLoaded', function() {
        const progressBars = document.querySelectorAll('.progress-bar');
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0';
            setTimeout(() => {
                bar.style.width = width;
            }, 500);
        });
    });
</script>
""", unsafe_allow_html=True)


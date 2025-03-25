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
import requests
import json
import re
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
from PIL import Image
from io import BytesIO
from streamlit_lottie import streamlit_lottie

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
        --primary-light: #A78BFA;
        --primary-dark: #6D28D9;
        --secondary-color: #4F46E5;
        --accent-color: #EC4899;
        --accent-light: #F9A8D4;
        --background-color: #F9FAFB;
        --card-bg-color: #FFFFFF;
        --text-color: #1F2937;
        --light-text: #6B7280;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --info-color: #3B82F6;
    }
    
    /* Dark mode colors */
    .dark {
        --primary-color: #8B5CF6;
        --primary-light: #A78BFA;
        --primary-dark: #7C3AED;
        --secondary-color: #6366F1;
        --accent-color: #F472B6;
        --accent-light: #F9A8D4;
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
    
    .badge-info {
        background-color: var(--info-color);
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
    
    .custom-button-secondary {
        background: transparent;
        color: var(--primary-color);
        border: 2px solid var(--primary-color);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .custom-button-secondary:hover {
        background-color: rgba(124, 58, 237, 0.1);
        transform: translateY(-2px);
    }
    
    /* Divider styling */
    .divider {
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        margin: 2rem 0;
        border-radius: 3px;
    }
    
    .divider-light {
        height: 1px;
        background: rgba(124, 58, 237, 0.2);
        margin: 1.5rem 0;
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
    
    /* Discover tab specific styling */
    .discover-container {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05), rgba(236, 72, 153, 0.05));
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .discover-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .discover-subheader {
        font-size: 1.2rem;
        color: var(--light-text);
        margin-bottom: 2rem;
    }
    
    .discover-card {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(124, 58, 237, 0.1);
    }
    
    .discover-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        border-color: var(--primary-color);
    }
    
    /* Tab styling */
    .custom-tabs {
        display: flex;
        border-bottom: 2px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    
    .custom-tab {
        padding: 1rem 2rem;
        cursor: pointer;
        font-weight: 600;
        color: var(--light-text);
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .custom-tab.active {
        color: var(--primary-color);
        border-bottom-color: var(--primary-color);
    }
    
    .custom-tab:hover:not(.active) {
        color: var(--text-color);
        border-bottom-color: #E5E7EB;
    }
    
    /* Recommendation card */
    .recommendation-card {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border-left: 5px solid var(--primary-color);
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05), rgba(236, 72, 153, 0.05));
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .feature-highlight-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--primary-color);
    }
    
    /* Animated counter */
    .counter-container {
        text-align: center;
        padding: 2rem;
    }
    
    .counter-value {
        font-size: 3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .counter-label {
        font-size: 1.2rem;
        color: var(--light-text);
    }
    
    /* Testimonial card */
    .testimonial-card {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        position: relative;
    }
    
    .testimonial-quote {
        font-size: 4rem;
        position: absolute;
        top: -20px;
        left: 20px;
        color: rgba(124, 58, 237, 0.1);
    }
    
    .testimonial-text {
        font-style: italic;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .testimonial-author {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Onboarding specific styling */
    .onboarding-container {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.03), rgba(236, 72, 153, 0.03));
        border-radius: 24px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow:72,153,0.03));
        border-radius: 24px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.05);
    }
    
    .onboarding-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .onboarding-subheader {
        font-size: 1.4rem;
        color: var(--light-text);
        margin-bottom: 2.5rem;
        text-align: center;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .onboarding-step {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary-color);
    }
    
    .onboarding-step:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .onboarding-step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        color: white;
        border-radius: 50%;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .onboarding-step-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .onboarding-step-description {
        color: var(--light-text);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .onboarding-cta {
        text-align: center;
        margin: 3rem 0 1rem;
    }
    
    .onboarding-feature-card {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border-top: 4px solid var(--primary-color);
    }
    
    .onboarding-feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    .onboarding-feature-icon {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .onboarding-feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }
    
    .onboarding-feature-description {
        color: var(--light-text);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* New Discover Tab Styling */
    .discover-new-container {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.02), rgba(236, 72, 153, 0.02));
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .discover-new-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .discover-new-subheader {
        font-size: 1.2rem;
        color: var(--light-text);
        margin-bottom: 2rem;
        text-align: center;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .discover-filter-container {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .discover-filter-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .discover-filter-section {
        margin-bottom: 1.5rem;
    }
    
    .discover-filter-section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-color);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .discover-results-container {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
    }
    
    .discover-results-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: var(--primary-color);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .discover-product-card {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
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
    
    .discover-product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }
    
    .discover-match-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        z-index: 10;
    }
    
    .discover-product-image-container {
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
    
    .discover-product-image {
        max-width: 100%;
        max-height: 100%;
        transition: transform 0.3s ease;
    }
    
    .discover-product-card:hover .discover-product-image {
        transform: scale(1.05);
    }
    
    .discover-product-title {
        font-size: 1.2rem;
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
    
    .discover-product-brand {
        color: var(--primary-color);
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .discover-product-meta {
        display: flex;
        justify-content: space-between;
        margin: 0.8rem 0;
    }
    
    .discover-product-description {
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
    
    .discover-product-features {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .discover-product-feature {
        background-color: rgba(124, 58, 237, 0.1);
        color: var(--primary-color);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .discover-product-footer {
        margin-top: auto;
        padding-top: 1rem;
        display: flex;
        gap: 1rem;
    }
    
    .discover-selected-product {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05), rgba(236, 72, 153, 0.05));
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-left: 5px solid var(--primary-color);
    }
    
    .discover-selected-product-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .discover-visualization-container {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
    }
    
    .discover-visualization-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }
    
    .discover-tabs {
        display: flex;
        border-bottom: 2px solid rgba(124, 58, 237, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .discover-tab {
        padding: 0.8rem 1.5rem;
        cursor: pointer;
        font-weight: 600;
        color: var(--light-text);
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
    }
    
    .discover-tab.active {
        color: var(--primary-color);
        border-bottom-color: var(--primary-color);
    }
    
    .discover-tab:hover:not(.active) {
        color: var(--text-color);
        border-bottom-color: rgba(124, 58, 237, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {e}")
        # Return a simple fallback animation
        return {
            "v": "5.5.7",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 200,
            "h": 200,
            "nm": "Fallback Animation",
            "ddd": 0,
            "assets": [],
            "layers": [{
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": "Circle",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [100, 100, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {
                        "a": 1,
                        "k": [
                            {"t": 0, "s": [100, 100, 100]},
                            {"t": 30, "s": [120, 120, 100]},
                            {"t": 60, "s": [100, 100, 100]}
                        ]
                    }
                },
                "shapes": [{
                    "ty": "el",
                    "p": {"a": 0, "k": [0, 0]},
                    "s": {"a": 0, "k": [80, 80]},
                    "d": 1,
                    "nm": "Ellipse Path 1",
                    "hd": false
                }],
                "style": {
                    "fill": {"a": 0, "k": [0.5, 0.2, 0.8]},
                    "stroke": {"a": 0, "k": [0.8, 0.4, 1]},
                    "strokeWidth": {"a": 0, "k": 4}
                }
            }]
        }

# Load animations with error handling
try:
    headphone_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qdchrpae.json")
    search_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qdchrpae.json")
    compare_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qdchrpae.json")
    onboarding_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qdchrpae.json")
except Exception as e:
    st.error(f"Error loading animations: {e}")
    # Create fallback animations
    headphone_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qdchrpae.json")
    search_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qdchrpae.json")
    compare_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qdchrpae.json")
    onboarding_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qdchrpae.json")

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

if 'onboarding_complete' not in st.session_state:
    st.session_state.onboarding_complete = False

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "price"

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

# Function to complete onboarding
def complete_onboarding():
    st.session_state.onboarding_complete = True

# Function to set active tab
def set_active_tab(tab):
    st.session_state.active_tab = tab

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
        df[['price_normalized', 'rating_normalized', 'reviews_normalized']] = scaler.fit_transform(df[['price', 'rating', 'reviews]])
        
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
        df[['price_normalized', 'rating_normalized', 'reviews_normalized']] = scaler.fit_transform(df[['price', 'rating', 'reviews]])
        
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
                'loyaltypoints': current_product['loyaltypoints],
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

# Onboarding Landing Page
elif not st.session_state.onboarding_complete:
    st.markdown('<div class="onboarding-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="onboarding-header">Welcome to SoundMatch</h1>', unsafe_allow_html=True)
    st.markdown('<p class="onboarding-subheader">Your AI-powered headphone recommendation system that helps you find the perfect match for your audio needs.</p>', unsafe_allow_html=True)
    
    # Lottie animation
    try:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            streamlit_lottie(onboarding_animation, height=250, key="onboarding_animation")
    except Exception as e:
        st.error(f"Error displaying animation: {e}")
        st.image("https://img.icons8.com/color/96/000000/headphones.png", width=200)
    
    # How it works section
    st.markdown('<h2 style="text-align: center; margin: 2rem 0 1rem;">How SoundMatch Works</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="onboarding-feature-card">
            <div class="onboarding-feature-icon">🔍</div>
            <h3 class="onboarding-feature-title">Smart Recommendations</h3>
            <p class="onboarding-feature-description">Our AI analyzes thousands of headphones to find the perfect match based on your preferences and needs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="onboarding-feature-card">
            <div class="onboarding-feature-icon">📊</div>
            <h3 class="onboarding-feature-title">Detailed Comparisons</h3>
            <p class="onboarding-feature-description">Compare headphones side by side with detailed specifications, features, and performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="onboarding-feature-card">
            <div class="onboarding-feature-icon">📱</div>
            <h3 class="onboarding-feature-title">Interactive Experience</h3>
            <p class="onboarding-feature-description">Explore visualizations, filter options, and detailed product information in an intuitive interface.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started steps
    st.markdown('<h2 style="text-align: center; margin: 3rem 0 1.5rem;">Getting Started</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="onboarding-step">
        <div class="onboarding-step-number">1</div>
        <h3 class="onboarding-step-title">Select a Headphone You Like</h3>
        <p class="onboarding-step-description">Start by selecting a headphone you already like or are interested in. Our AI will use this as a reference point to find similar products that match your taste.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="onboarding-step">
        <div class="onboarding-step-number">2</div>
        <h3 class="onboarding-step-title">Customize Your Preferences</h3>
        <p class="onboarding-step-description">Adjust filters like price range, brand, connectivity type, and headphone style to narrow down the recommendations to exactly what you're looking for.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="onboarding-step">
        <div class="onboarding-step-number">3</div>
        <h3 class="onboarding-step-title">Explore Recommendations</h3>
        <p class="onboarding-step-description">Review your personalized recommendations, compare features, check match scores, and visualize differences to make an informed decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown('<div class="onboarding-cta">', unsafe_allow_html=True)
    if st.button("Get Started", type="primary", key="start_button"):
        complete_onboarding()
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Discover tab - Completely Redesigned
elif selected == "Discover":
    # Main container for the discover tab
    st.markdown('<div class="discover-new-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="discover-new-header">Find Your Perfect Sound</h1>', unsafe_allow_html=True)
    st.markdown('<p class="discover-new-subheader">Our AI-powered recommendation engine analyzes thousands of headphones to find your perfect match based on your preferences and listening habits.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Filters in a stylable container
        st.markdown('<div class="discover-filter-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="discover-filter-title">🎧 Personalize Your Search</h3>', unsafe_allow_html=True)
        
        # Product selection
        st.markdown('<p class="discover-filter-section-title">🔍 Select a Reference Product</p>', unsafe_allow_html=True)
        product_name = st.selectbox(
            "Choose a headphone you like:",
            df['name'].tolist(),
            index=0,
            help="This will be used as a reference to find similar products"
        )
        
        # Number of recommendations
        st.markdown('<p class="discover-filter-section-title">📊 Number of Results</p>', unsafe_allow_html=True)
        top_n = st.slider(
            "How many recommendations?",
            min_value=1,
            max_value=6,
            value=3,
            help="Select how many product recommendations you want to see"
        )
        
        # Price range filter
        st.markdown('<p class="discover-filter-section-title">💰 Price Range</p>', unsafe_allow_html=True)
        min_price = int(df['price'].min())
        max_price = int(df['price'].max())
        price_range = st.slider(
            "Select your budget (₹)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            help="Filter products within your budget"
        )
        
        # Rating filter
        st.markdown('<p class="discover-filter-section-title">⭐ Minimum Rating</p>', unsafe_allow_html=True)
        min_rating = st.slider(
            "Select minimum rating",
            min_value=3.0,
            max_value=5.0,
            value=3.5,
            step=0.1,
            help="Only show products with ratings at or above this value"
        )
        
        st.markdown('<div class="divider-light"></div>', unsafe_allow_html=True)
        
        # Advanced filters
        st.markdown('<p class="discover-filter-section-title">🔧 Advanced Filters</p>', unsafe_allow_html=True)
        
        # Brand filter
        brand_options = ["Any"] + sorted(df['brand'].unique().tolist())
        brand = st.selectbox(
            "Brand preference",
            brand_options,
            help="Filter by specific brands"
        )
        brand_filter = None if brand == "Any" else brand
        
        # Connectivity filter
        connectivity = st.radio(
            "Connectivity type",
            options=["Any", "Wireless", "Wired"],
            horizontal=True,
            help="Choose between wireless or wired headphones"
        )
        connectivity_filter = None if connectivity == "Any" else connectivity
        
        # Headphone type filter
        headphone_type = st.radio(
            "Headphone style",
            options=["Any", "Over-Ear", "On-Ear", "In-Ear"],
            horizontal=True,
            help="Select the style of headphones you prefer"
        )
        type_filter = None if headphone_type == "Any" else headphone_type
        
        # Find button
        find_button = st.button(
            "Find My Perfect Match",
            type="primary",
            use_container_width=True
        )
        
        # Reset button
        reset_button = st.button(
            "Reset Filters",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Lottie animation
        try:
            streamlit_lottie(headphone_animation, height=200, key="headphone_animation")
        except Exception as e:
            st.error(f"Error displaying animation: {e}")
            st.image("https://img.icons8.com/color/96/000000/headphones.png", width=200)
        
        # Quick tips
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - **Start with what you know**: Select a product you already like as a starting point
            - **Set your budget**: Adjust price range to find alternatives in your budget
            - **Consider your usage**: Use the connectivity filter to find specific types of headphones
            - **Check the match score**: Higher similarity scores indicate closer matches to your selected product
            - **Compare features**: Look at battery life for wireless options and sound quality metrics
            """)
    
    with col2:
        if find_button:
            with st.spinner("Finding your perfect matches..."):
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
                st.markdown('<div class="discover-selected-product">', unsafe_allow_html=True)
                st.markdown('<h3 class="discover-selected-product-title">Your Selected Product</h3>', unsafe_allow_html=True)
                selected_product = df[df['name'] == product_name].iloc[0]
                
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
                    <div style="margin: 1rem 0;">
                        <span class="badge badge-primary">{selected_product['type']}</span>
                        <span class="badge badge-secondary">{selected_product['connectivity']}</span>
                        {f'<span class="badge badge-accent">{selected_product["battery_life"]}h Battery</span>' if selected_product['battery_life'] > 0 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # View details button
                    if st.button("View Details", key="view_selected_details"):
                        show_product_detail(selected_product['name'])
                        st.experimental_rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display recommendations
                st.markdown('<div class="discover-results-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="discover-results-header">🎯 Your Personalized Recommendations</h3>', unsafe_allow_html=True)
                
                # Create a grid for recommendations
                for i, rec in enumerate(recommendations):
                    # Calculate match percentage
                    match_percentage = int(rec['similarity'] * 100)
                    
                    # Extract key features
                    features = extract_features(rec['description'])
                    key_features = features[:3] if features else []
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"""
                        <div class="discover-product-card">
                            <div class="discover-match-badge">{match_percentage}% Match</div>
                            <div class="discover-product-image-container">
                                <img src="{rec['image_url']}" class="discover-product-image" alt="{rec['name']}">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <h3 class="discover-product-title">{rec['name']}</h3>
                        <p class="discover-product-brand">By {rec['brand']}</p>
                        
                        <div class="discover-product-meta">
                            <span class="price-tag">₹{rec['price']}</span>
                            <span class="rating-tag">★ {rec['rating']} ({rec['reviews']:,} reviews)</span>
                        </div>
                        
                        <div style="margin: 0.5rem 0;">
                            <span class="badge badge-primary">{rec['type']}</span>
                            <span class="badge badge-secondary">{rec['connectivity']}</span>
                            {f'<span class="badge badge-accent">{rec["battery_life"]}h Battery</span>' if rec['battery_life'] > 0 else ''}
                        </div>
                        
                        <p class="discover-product-description">{rec['description']}</p>
                        """, unsafe_allow_html=True)
                        
                        if key_features:
                            st.markdown('<div class="discover-product-features">', unsafe_allow_html=True)
                            for feature in key_features:
                                st.markdown(f'<span class="discover-product-feature">{feature}</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("View Details", key=f"view_details_{i}"):
                                show_product_detail(rec['name'])
                                st.experimental_rerun()
                        with col_b:
                            st.button("Compare", key=f"compare_{i}")
                    
                    if i < len(recommendations) - 1:
                        st.markdown('<div class="divider-light"></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization of recommendations
                st.markdown('<div class="discover-visualization-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="discover-visualization-title">📊 Recommendation Analysis</h3>', unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                st.markdown('<div class="discover-tabs">', unsafe_allow_html=True)
                
                # Tab buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    price_tab_class = "active" if st.session_state.active_tab == "price" else ""
                    if st.button("Price Comparison", key="price_tab", use_container_width=True):
                        set_active_tab("price")
                        st.experimental_rerun()
                
                with col2:
                    match_tab_class = "active" if st.session_state.active_tab == "match" else ""
                    if st.button("Match Score", key="match_tab", use_container_width=True):
                        set_active_tab("match")
                        st.experimental_rerun()
                
                with col3:
                    radar_tab_class = "active" if st.session_state.active_tab == "radar" else ""
                    if st.button("Feature Radar", key="radar_tab", use_container_width=True):
                        set_active_tab("radar")
                        st.experimental_rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Tab content
                if st.session_state.active_tab == "price":
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
                
                elif st.session_state.active_tab == "match":
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
                
                else:  # Radar tab
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
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Display error message if no recommendations found
                st.markdown("""
                <div class="error-container animate-fade-in">
                    <div class="error-icon">😕</div>
                    <h2 class="error-title">No Recommendations Found</h2>
                    <p class="error-message">We couldn't find any products that match your filters. Try adjusting your criteria to see more results.</p>
                    <button class="custom-button" onclick="resetFilters()">Reset Filters</button>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Display welcome message and instructions with enhanced design
            st.markdown("""
            <div class="discover-card animate-fade-in" style="text-align: center; padding: 50px 20px; margin-top: 20px;">
                <img src="https://img.icons8.com/color/96/000000/headphones.png" width="100">
                <h2 style="margin-top: 20px; font-size: 1.8rem; background: linear-gradient(90deg, var(--primary-color), var(--accent-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Find Your Perfect Sound Match</h2>
                <p style="margin: 20px 0; font-size: 1.2rem;">
                    Our AI-powered recommendation system analyzes thousands of headphones to find your perfect match.
                </p>
                <div class="feature-highlight">
                    <p class="feature-highlight-title">How It Works:</p>
                    <ol style="text-align: left; padding-left: 20px;">
                        <li>Select a headphone you already like or are interested in</li>
                        <li>Adjust filters to match your preferences and budget</li>
                        <li>Click "Find My Perfect Match" to see personalized recommendations</li>
                        <li>Compare features, prices, and match scores to make your decision</li>
                    </ol>
                </div>
                <p style="margin-top: 20px;">Ready to discover your perfect headphones? Start by selecting a product on the left.</p>
            </div>
            """, unsafe_allow_html=True)

# Compare tab, Explore tab, Insights tab, and About tab code would continue here...

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
    
    // Reset filters function
    function resetFilters() {
        // This will be handled by the Streamlit reset button
        document.querySelector('button[data-testid="baseButton-secondary"]').click();
    }
</script>
""", unsafe_allow_html=True)


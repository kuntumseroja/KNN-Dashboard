# KNN Banking Account Relationship Dashboard

## Overview
An interactive Streamlit dashboard that demonstrates K-Nearest Neighbors (KNN) algorithm for detecting behavioral relationships between banking accounts. The application identifies implicit connections between accounts based on transaction patterns, device usage, and behavioral features.

## Current State
- Fully functional MVP with mock banking data generation
- KNN-based similarity detection with configurable parameters
- Interactive network graph visualization
- Community detection using Louvain algorithm
- Account lookup with neighbor comparison

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with all dashboard components
- `.streamlit/config.toml` - Streamlit server configuration

### Key Features
1. **Mock Data Generation**: Creates realistic banking account data with features like:
   - Transaction amounts and counts
   - Salary inflows
   - Cash withdrawal ratios
   - Device and IP diversity
   - Country risk scores
   - Segment codes (Retail/SME/Corporate)

2. **KNN Similarity Engine**: 
   - Uses scikit-learn's NearestNeighbors with preprocessing pipeline
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
   - Euclidean distance metric

3. **Network Visualization**: 
   - Interactive Plotly graph showing account relationships
   - Color-coded by community membership
   - Node size based on connection count

4. **Community Detection**: 
   - Uses NetworkX Louvain algorithm
   - Identifies potential fraud rings, households, or business groups

### Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- networkx
- plotly

## Recent Changes
- December 2025: Initial MVP created with all core features

## User Preferences
- None specified yet

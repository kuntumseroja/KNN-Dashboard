# KNN Banking Account Relationship Dashboard

## Overview
An interactive Streamlit dashboard that demonstrates K-Nearest Neighbors (KNN) algorithm for detecting behavioral relationships between banking accounts. The application identifies implicit connections between accounts based on transaction patterns, device usage, and behavioral features.

## Current State
- Fully functional application with mock banking data generation
- CSV/Excel file upload support for custom datasets
- KNN-based similarity detection with configurable parameters
- Interactive network graph visualization
- Multiple clustering algorithms (Louvain and DBSCAN)
- Fraud risk scoring system
- Export functionality for data and analysis results

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with all dashboard components
- `.streamlit/config.toml` - Streamlit server configuration

### Key Features

1. **Data Sources**:
   - Mock data generation with realistic banking account features
   - CSV/Excel file upload with validation
   - Downloadable template for custom data

2. **KNN Similarity Engine**: 
   - Uses scikit-learn's NearestNeighbors with preprocessing pipeline
   - StandardScaler for numeric features
   - OneHotEncoder for categorical features
   - Configurable K neighbors and distance threshold

3. **Network Visualization**: 
   - Interactive Plotly graph showing account relationships
   - Color-coded by community or risk score
   - Node size based on connection count
   - Hover details with account metrics

4. **Clustering Algorithms**: 
   - Louvain: Graph-based community detection
   - DBSCAN: Density-based clustering in feature space
   - Side-by-side comparison view

5. **Fraud Risk Scoring**:
   - Hub score (network centrality)
   - Bridge score (betweenness centrality)
   - Cluster score (community size)
   - Behavioral score (cash withdrawals, country risk, device diversity)
   - Risk levels: High, Medium, Low

6. **Export Functionality**:
   - Export accounts data as CSV
   - Export relationships as CSV
   - Export risk scores as CSV

### Dashboard Tabs
1. **Network Overview**: Graph visualization, metrics, hub accounts
2. **Account Lookup**: Individual account details, neighbors, explanations
3. **Community Analysis**: Community statistics, segment distribution
4. **Fraud Risk Analysis**: Risk metrics, distributions, high-risk accounts
5. **Clustering Comparison**: Louvain vs DBSCAN comparison
6. **Data Explorer**: Raw data tables, correlations, exports

### Account Features Used
- avg_txn_amount_30d - Average transaction amount (30 days)
- txn_count_30d - Transaction count (30 days)
- salary_inflow_90d - Salary inflow (90 days)
- cash_withdrawal_ratio_90d - Cash withdrawal ratio (90 days)
- merchant_diversity_90d - Merchant diversity (90 days)
- unique_devices_90d - Unique devices (90 days)
- unique_ips_90d - Unique IPs (90 days)
- country_risk_score - Country risk score
- channel_mix_online_ratio - Online channel usage ratio
- segment_code - Account segment (Retail/SME/Corporate)

### Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- networkx
- plotly
- openpyxl (for Excel file support)

## Recent Changes
- December 2025: Added CSV/Excel upload, DBSCAN clustering, fraud risk scoring, exports
- December 2025: Initial MVP created with KNN, Louvain, network visualization

## User Preferences
- None specified yet

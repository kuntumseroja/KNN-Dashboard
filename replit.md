# OBS Account Relationship Dashboard

## Overview
An interactive Streamlit dashboard for Relationship Managers (RM) that discovers behavioral relationships between accounts in the poultry supply chain ecosystem using K-Nearest Neighbors (KNN) algorithm. The dashboard focuses on the JAPFA anchor group and related ecosystem players, helping RMs identify opportunities and understand account networks.

## Current State
- Fully functional dashboard customized for poultry ecosystem
- Poultry-specific mock data generator with realistic supply chain roles
- KNN-based similarity detection with configurable parameters
- Interactive network visualization with ecosystem role color coding
- RM Opportunity scoring system (replaces fraud risk scoring)
- Community detection using Louvain and DBSCAN algorithms
- CSV/Excel upload support for custom datasets
- Export functionality for accounts, relationships, and opportunities
- **Cross-Anchor Relationship Detection**: Identify relationships between accounts across different anchor groups
- **Bridge Account Analysis**: Find accounts connecting multiple anchor ecosystems
- **Cross-Anchor Opportunity Scoring**: Prioritize high-value cross-anchor relationships

## Project Architecture

### Main Files
- `app.py` - Main Streamlit application with all dashboard components
- `.streamlit/config.toml` - Streamlit server configuration

### Poultry Ecosystem Model

#### Ecosystem Roles
- **Anchor_Corporate** - Large corporate anchors (e.g., Japfa Comfeed, Charoen Pokphand)
- **Feed_Mill** - Regional feed production facilities
- **Breeder_Farm** - Parent stock and breeding operations
- **Contract_Farmer** - Partner farms under anchor contracts
- **Collector/Offtaker** - Livebird aggregators and buyers
- **Slaughterhouse** - Processing facilities (RPH)
- **Retailer/Warung** - End retail and traditional markets
- **Logistics/Transport** - Cold chain and transportation
- **Input_Supplier** - Veterinary, feed additives, equipment

#### Anchor Groups
- Japfa_Group
- CP_Group
- Charoen_Group
- Malindo_Group
- Independent

#### Account Features
- account_id - Unique account identifier
- legal_name - Company/business legal name
- ecosystem_role - Role in poultry supply chain
- anchor_group - Parent anchor affiliation
- anchor_level - Distance from anchor (0=anchor, 1=direct, 2=indirect)
- segment_code - Corporate/SME/Micro
- primary_bank - Current primary bank
- bri_status - BRI banking status (Existing/NTB)
- ntb_status - New-to-Bank status
- avg_txn_amount_30d - Average transaction amount (30 days)
- txn_count_30d - Transaction count (30 days)
- turnover_90d - Total turnover (90 days)
- cash_withdrawal_ratio_90d - Cash withdrawal ratio
- merchant_diversity_90d - Number of unique merchants
- unique_devices_90d - Device fingerprint diversity
- unique_ips_90d - IP address diversity
- country_risk_score - Geographic risk indicator
- channel_mix_online_ratio - Digital channel usage
- lead_score_bri - BRI lead scoring (0-100)

### Dashboard Tabs

1. **Ecosystem Overview**: Network visualization, composition charts, hub accounts, cross-anchor metrics
2. **Account Details**: Individual account lookup with connected accounts, opportunity scores, cross-anchor relationship details
3. **Anchor Analysis**: Anchor group statistics, turnover analysis, supply chain sunburst, **cross-anchor relationship analysis**
4. **RM Opportunities**: Priority pipeline, NTB targets, opportunity scoring methodology
5. **Cluster Analysis**: Louvain vs DBSCAN comparison, community exploration
6. **Data Export**: Download datasets, view correlations

### Opportunity Scoring System
The opportunity score (0-100) is calculated based on:
- **Network Score (20%)**: Account connectivity in the ecosystem network
- **Influence Score (15%)**: Betweenness centrality (bridging communities)
- **Ecosystem Score (15%)**: Community size and strength
- **Lead Score (25%)**: BRI lead scoring based on financial behavior
- **Potential Score (25%)**: NTB status bonus and anchor proximity

Priority levels:
- High: Score >= 70
- Medium: Score >= 45
- Low: Score < 45

### Key Technologies
- **KNN Engine**: scikit-learn NearestNeighbors with preprocessing pipeline
- **Network Analysis**: NetworkX for graph operations and community detection
- **Visualization**: Plotly for interactive charts and network graphs
- **Clustering**: Louvain (graph-based) and DBSCAN (density-based)

### Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- networkx
- plotly
- openpyxl (Excel support)

## Recent Changes
- January 2025: Added Cross-Anchor Relationship Detection with comprehensive parameters and metrics
- January 2025: Implemented Bridge Account identification and scoring system
- January 2025: Enhanced network visualization with cross-anchor edge highlighting
- January 2025: Added Cross-Anchor Opportunity scoring and analysis dashboard
- December 2025: Rebranded to "OBS Account Relationship Dashboard"
- December 2025: Implemented poultry ecosystem data model with JAPFA as anchor
- December 2025: Modernized UI for Relationship Manager focus
- December 2025: Added RM Opportunity scoring replacing fraud risk scoring
- December 2025: Added Anchor Analysis and ecosystem role visualization
- December 2025: Initial MVP with KNN, community detection, network visualization

## Documentation
- **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)**: Comprehensive documentation on cross-anchor relationship detection parameters, use cases, and best practices

## User Preferences
- Dashboard focused on RM use case for banking opportunities
- Poultry supply chain ecosystem context
- Indonesian business naming conventions (PT, CV, UD, Koperasi)
- Currency display in Rupiah (Rp) with B/M/K abbreviations

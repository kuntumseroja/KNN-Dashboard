# OBS Account Relationship Dashboard

## Overview
An interactive Streamlit dashboard for Relationship Managers (RM) that discovers behavioral relationships between accounts in the poultry supply chain ecosystem using K-Nearest Neighbors (KNN) algorithm. The dashboard focuses on the JAPFA anchor group and related ecosystem players, helping RMs identify opportunities and understand account networks.

## Current State
- Fully functional dashboard customized for poultry ecosystem
- Poultry-specific mock data generator with realistic supply chain roles
- **KNN-based relationship identification**: Discovers behavioral relationships between accounts using similarity analysis
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

**For detailed architecture documentation, see: [System Architecture Guide](ARCHITECTURE.md)**

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

### Relationship Identification

The dashboard identifies relationships between accounts using **K-Nearest Neighbors (KNN)** algorithm:

- **Behavioral Similarity**: Accounts with similar transaction patterns are identified as related
- **Feature-Based**: Uses 10 numeric features + 2 categorical features for similarity calculation
- **Network Construction**: Relationships are built into a graph for network analysis
- **Community Detection**: Accounts are grouped into communities using clustering algorithms
- **Cross-Anchor Detection**: Special handling for relationships across different anchor groups

**Key Parameters**:
- **K Neighbors**: Number of similar accounts to find (default: 5, range: 3-12)
- **Similarity Threshold**: Maximum distance for relationship (default: 2.0, range: 0.5-4.0)

See **[Ecosystem Relationship Identification Guide](ECOSYSTEM_RELATIONSHIP_IDENTIFICATION.md)** for complete documentation.

### Opportunity Scoring System
The opportunity score (0-100) is calculated based on:
- **Network Score (20 points)**: Account connectivity in the ecosystem network (degree centrality)
- **Influence Score (15 points)**: Betweenness centrality (bridging communities)
- **Ecosystem Score (15 points)**: Community size and strength
- **Lead Score (25 points)**: BRI lead scoring based on financial behavior
- **Potential Score (25 points)**: NTB status bonus (+15) and anchor proximity (+10)

Priority levels:
- **High**: Score ≥ 70 (Top-quality leads, immediate RM action)
- **Medium**: Score 45-69 (Good opportunities, RM follow-up)
- **Low**: Score < 45 (Standard accounts, routine monitoring)

See **[Lead & Opportunity Identification Guide](LEAD_OPPORTUNITY_IDENTIFICATION.md)** for comprehensive documentation on identifying quality leads and interpreting opportunity scores.

### Key Technologies
- **KNN Engine**: scikit-learn NearestNeighbors with preprocessing pipeline
- **Network Analysis**: NetworkX for graph operations and community detection
- **Visualization**: Plotly for interactive charts and network graphs
- **Clustering**: Louvain (graph-based) and DBSCAN (density-based)
- **NER & Anonymization**: spaCy (optional) or regex-based entity detection for data privacy compliance

### Dependencies
- streamlit
- pandas
- numpy
- scikit-learn
- networkx
- plotly
- openpyxl (Excel support)
- spacy (optional, for enhanced NER-based anonymization)

## Data Privacy & Anonymization

### UU PDP Compliance
The dashboard implements **automatic NER-based anonymization** to comply with:
- **UU PDP** (Undang-Undang Perlindungan Data Pribadi) - Indonesian Personal Data Protection Law
- **Banking Confidentiality** requirements (Kerahasiaan Customer)

### How It Works
- **Automatic Anonymization**: All customer data is automatically anonymized before display
- **NER-Based Detection**: Uses Named Entity Recognition (NER) to identify and mask:
  - Company names (PT, CV, UD, Koperasi, etc.)
  - Person names
  - Other sensitive organizational information
- **Consistent Mapping**: Same entities receive the same anonymized identifier throughout the session
- **Dual-Mode Operation**:
  - **Primary**: spaCy NER model (if installed) for high accuracy
  - **Fallback**: Regex-based detection for Indonesian business patterns

### Anonymization Features
- **Entity Types Detected**: PERSON, ORG (Organization)
- **Anonymization Format**: 
  - Persons: `PERSON_[8-char-hash]`
  - Organizations: `PT ENTITY_[8-char-hash]` (preserves company type prefix)
- **Fields Anonymized**: `legal_name` (automatically applied to all data)
- **Session Persistence**: Anonymization mapping maintained across dashboard interactions

### Installation (Optional)
For enhanced accuracy, install spaCy with Indonesian language model:
```bash
pip install spacy
python -m spacy download id_core_web_sm    # Indonesian (recommended)
# OR
python -m spacy download xx_ent_wiki_sm    # Multilingual
# OR
python -m spacy download en_core_web_sm    # English
```

The dashboard works without spaCy using regex-based anonymization, but NER models provide better accuracy.

### Privacy Notice
All data displayed in the dashboard is automatically anonymized. The anonymization cannot be disabled to ensure compliance with data protection regulations and banking confidentiality requirements.

## Recent Changes
- January 2025: **Added NER-based anonymization** for UU PDP and banking confidentiality compliance
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
- **[System Architecture Guide](ARCHITECTURE.md)**: Comprehensive architecture documentation covering high-level system design, data architecture, model architecture, training flow, and inference flow
- **[Lead & Opportunity Identification Guide](LEAD_OPPORTUNITY_IDENTIFICATION.md)**: Comprehensive guide on identifying quality leads, interpreting opportunity scores, prioritizing accounts, and best practices for RM actions
- **[Ecosystem Relationship Identification Guide](ECOSYSTEM_RELATIONSHIP_IDENTIFICATION.md)**: Comprehensive guide on how KNN identifies behavioral relationships between accounts, parameter tuning, use cases, and best practices
- **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)**: Comprehensive documentation on cross-anchor relationship detection parameters, use cases, and best practices
- **[Data Privacy & Anonymization Guide](DATA_PRIVACY_ANONYMIZATION.md)**: Complete documentation on NER-based anonymization, UU PDP compliance, installation, and troubleshooting

## User Preferences
- Dashboard focused on RM use case for banking opportunities
- Poultry supply chain ecosystem context
- Indonesian business naming conventions (PT, CV, UD, Koperasi)
- Currency display in Rupiah (Rp) with B/M/K abbreviations


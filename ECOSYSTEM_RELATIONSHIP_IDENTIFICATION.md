# Ecosystem Relationship Identification Guide

## Overview

The OBS Account Relationship Dashboard uses **K-Nearest Neighbors (KNN)** algorithm to identify behavioral relationships between accounts within the poultry supply chain ecosystem. This guide explains how the system discovers implicit relationships based on transaction patterns, business behavior, and operational characteristics.

## What are Ecosystem Relationships?

Ecosystem relationships are **behavioral connections** between accounts that share similar:
- Transaction patterns
- Business characteristics
- Operational behaviors
- Financial profiles

Unlike explicit relationships (same customer ID, same business group), these are **implicit relationships** discovered through machine learning analysis of account behavior.

### Why Identify Ecosystem Relationships?

1. **Opportunity Discovery**: Find accounts with similar profiles for cross-selling
2. **Network Analysis**: Understand ecosystem structure and connections
3. **Risk Management**: Detect unusual relationship patterns
4. **Market Intelligence**: Identify business clusters and supply chain connections
5. **NTB Acquisition**: Target accounts similar to high-value existing customers

## How KNN Identifies Relationships

### Step 1: Feature Engineering

Each account is represented as a **feature vector** containing:

#### Numeric Features (10 features)
- `avg_txn_amount_30d` - Average transaction amount (30 days)
- `txn_count_30d` - Transaction count (30 days)
- `turnover_90d` - Total turnover (90 days)
- `cash_withdrawal_ratio_90d` - Cash withdrawal ratio
- `merchant_diversity_90d` - Number of unique merchants
- `unique_devices_90d` - Device fingerprint diversity
- `unique_ips_90d` - IP address diversity
- `country_risk_score` - Geographic risk indicator
- `channel_mix_online_ratio` - Digital channel usage ratio
- `lead_score_bri` - BRI lead scoring (0-100)

#### Categorical Features (2 features)
- `segment_code` - Corporate/SME/Micro
- `ecosystem_role` - Role in supply chain (Anchor_Corporate, Feed_Mill, etc.)

### Step 2: Data Preprocessing

The system applies a preprocessing pipeline:

```python
1. Numeric Features → StandardScaler (normalization)
2. Categorical Features → OneHotEncoder (binary encoding)
3. Combined → ColumnTransformer (parallel processing)
```

**Purpose**: 
- Normalize numeric features to same scale (0-1 range)
- Encode categorical features as binary vectors
- Ensure all features contribute equally to similarity calculation

### Step 3: KNN Similarity Search

For each account, KNN finds the **K most similar accounts**:

```python
1. Calculate Euclidean distance to all other accounts
2. Select K nearest neighbors (default: 5)
3. Filter by distance threshold (default: 2.0)
4. Convert distance to similarity score
```

**Similarity Formula**:
```
similarity = exp(-distance)
```

- **Distance = 0** → Similarity = 1.0 (identical)
- **Distance = 1** → Similarity = 0.37 (very similar)
- **Distance = 2** → Similarity = 0.14 (moderately similar)
- **Distance > 2** → Filtered out (not similar enough)

### Step 4: Relationship Edge Creation

Each similar account pair becomes a **relationship edge**:

```python
Edge Properties:
- src: Source account ID
- dst: Destination account ID
- distance: Euclidean distance (lower = more similar)
- similarity: Similarity score 0-1 (higher = more similar)
- src_anchor: Source anchor group
- dst_anchor: Destination anchor group
- is_cross_anchor: Boolean (different anchor groups)
- is_anchor_bridge: Boolean (involves anchor-level account)
```

### Step 5: Network Graph Construction

Relationships are built into a **NetworkX graph**:

```python
Nodes = Accounts
Edges = Relationships (similarity > threshold)
```

This enables:
- **Community Detection**: Find account clusters
- **Centrality Analysis**: Identify hub accounts
- **Path Analysis**: Discover connection chains
- **Visualization**: Network graphs and relationship maps

## Key Parameters

### K Neighbors (`n_neighbors`)

**Default**: 5  
**Range**: 3-12  
**Location**: Sidebar → KNN Parameters

**Impact**:
- **Lower (3-4)**: Fewer relationships, higher precision
- **Higher (8-12)**: More relationships, broader network
- **Recommended**: 5-6 for balanced results

**Use Cases**:
- **Precision Focus**: Use K=3-4 for high-confidence relationships
- **Network Discovery**: Use K=8-10 to find more connections
- **Large Datasets**: Use K=6-8 for comprehensive coverage

### Similarity Threshold (`max_distance`)

**Default**: 2.0  
**Range**: 0.5-4.0  
**Location**: Sidebar → KNN Parameters

**Impact**:
- **Lower (0.5-1.5)**: Only very similar accounts (strong relationships)
- **Higher (2.5-4.0)**: More accounts included (broader network)
- **Recommended**: 2.0 for balanced results

**Distance → Similarity Mapping**:
| Distance | Similarity | Relationship Strength |
|----------|------------|----------------------|
| 0.0-0.5  | 0.61-1.0   | Very Strong          |
| 0.5-1.0  | 0.37-0.61  | Strong                |
| 1.0-1.5  | 0.22-0.37  | Moderate              |
| 1.5-2.0  | 0.14-0.22  | Weak                  |
| >2.0     | <0.14      | Filtered Out         |

**Use Cases**:
- **High Precision**: Threshold 1.5 (only strong relationships)
- **Balanced**: Threshold 2.0 (default, recommended)
- **Broad Network**: Threshold 2.5-3.0 (find more connections)

## Relationship Types

### 1. Intra-Anchor Relationships

**Definition**: Relationships between accounts in the **same anchor group**

**Characteristics**:
- `is_cross_anchor = False`
- `src_anchor == dst_anchor`
- Typically stronger similarity (same ecosystem)

**Use Cases**:
- **Supply Chain Mapping**: Understand relationships within anchor ecosystem
- **Product Cross-Sell**: Target similar accounts in same group
- **Risk Assessment**: Monitor relationship patterns within ecosystem

### 2. Cross-Anchor Relationships

**Definition**: Relationships between accounts in **different anchor groups**

**Characteristics**:
- `is_cross_anchor = True`
- `src_anchor != dst_anchor`
- Both anchors are not "Independent"

**Use Cases**:
- **Market Expansion**: Identify bridges between ecosystems
- **Strategic Partnerships**: Find collaboration opportunities
- **Competitive Intelligence**: Understand cross-ecosystem connections

See **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)** for detailed documentation.

### 3. Role-Based Relationships

**Definition**: Relationships between accounts with **same ecosystem role**

**Examples**:
- Contract_Farmer ↔ Contract_Farmer
- Retailer/Warung ↔ Retailer/Warung
- Feed_Mill ↔ Feed_Mill

**Characteristics**:
- Similar transaction patterns
- Similar business models
- Potential for peer comparison

**Use Cases**:
- **Benchmarking**: Compare accounts in same role
- **Best Practice Sharing**: Identify top performers
- **Market Segmentation**: Understand role-specific behaviors

### 4. Supply Chain Relationships

**Definition**: Relationships following **supply chain flow**

**Examples**:
- Feed_Mill → Contract_Farmer
- Contract_Farmer → Collector/Offtaker
- Slaughterhouse → Retailer/Warung

**Characteristics**:
- Complementary transaction patterns
- Upstream-downstream connections
- Business flow relationships

**Use Cases**:
- **Supply Chain Mapping**: Visualize ecosystem flow
- **Value Chain Analysis**: Understand business relationships
- **Partnership Opportunities**: Identify complementary businesses

## Dashboard Features

### Ecosystem Overview Tab

#### Network Visualization
- **Interactive Graph**: Nodes = accounts, Edges = relationships
- **Color Coding**: By ecosystem role, anchor group, or community
- **Node Size**: Based on connection count (hubs are larger)
- **Edge Thickness**: Based on similarity strength
- **Cross-Anchor Highlighting**: Red edges for cross-anchor relationships

#### Top Connected Accounts (Ecosystem Hubs)
- **Purpose**: Identify accounts with most relationships
- **Metrics**: Connection count, centrality score, lead score
- **Use Case**: Find influential accounts in ecosystem

### Account Details Tab

#### Connected Accounts
- **Purpose**: View all accounts related to selected account
- **Display**: Table with similarity scores, roles, anchor groups
- **Sorting**: By similarity (highest first)
- **Cross-Anchor Indicator**: Shows if relationship is cross-anchor

#### Relationship Insight
- **Purpose**: Explain why accounts are related
- **Content**: Similarity score, transaction pattern explanation
- **Use Case**: Understand relationship rationale

### Cluster Analysis Tab

#### Community Detection
- **Louvain Algorithm**: Graph-based community detection
- **DBSCAN Algorithm**: Density-based clustering
- **Purpose**: Find account clusters (communities)

#### Community Exploration
- **Community Details**: Accounts, roles, statistics
- **Visualization**: Role distribution, bank status
- **Use Case**: Understand ecosystem structure

## Use Cases

### 1. NTB Acquisition Targeting

**Scenario**: Find accounts similar to high-value existing customers

**Steps**:
1. Identify high-value existing customer (high lead_score, high turnover)
2. View connected accounts in Account Details tab
3. Filter for NTB accounts (`bri_status == "NTB"`)
4. Prioritize by similarity score and opportunity score

**Parameters**:
- Similarity threshold: 2.0 (default)
- K neighbors: 5-6
- Focus: High similarity (>0.3) + NTB status

### 2. Supply Chain Mapping

**Scenario**: Understand relationships in poultry supply chain

**Steps**:
1. Set similarity threshold to 2.5 (broader network)
2. View network visualization colored by ecosystem_role
3. Identify relationship patterns between roles
4. Analyze community structure

**Parameters**:
- Similarity threshold: 2.5 (broader network)
- K neighbors: 8-10 (more connections)
- Color by: Ecosystem Role

### 3. Risk Detection

**Scenario**: Find unusual relationship patterns

**Steps**:
1. Identify accounts with unexpected relationships
2. Look for cross-anchor relationships with low similarity
3. Check for accounts connecting distant anchor levels
4. Flag for review

**Indicators**:
- Low similarity (<0.2) but still connected
- Cross-anchor with large anchor_level_diff
- Unusual role combinations

### 4. Product Cross-Sell

**Scenario**: Identify accounts for product recommendations

**Steps**:
1. Select target account (existing customer with product X)
2. View connected accounts
3. Filter for accounts without product X
4. Prioritize by similarity and opportunity score

**Parameters**:
- Similarity threshold: 1.5-2.0 (strong relationships)
- Focus: High similarity (>0.3) + opportunity score

### 5. Ecosystem Hub Identification

**Scenario**: Find influential accounts in ecosystem

**Steps**:
1. View Ecosystem Overview tab
2. Check "Top Connected Accounts" section
3. Analyze hub accounts (high connection count)
4. Review their relationships and influence

**Metrics**:
- Degree centrality (connection count)
- Betweenness centrality (bridge importance)
- Community size (ecosystem influence)

## Best Practices

### 1. Parameter Tuning

**Start with Defaults**:
- K neighbors: 5
- Similarity threshold: 2.0

**Adjust Based on Results**:
- **Too Few Relationships**: Increase K (6-8) or threshold (2.5)
- **Too Many Weak Relationships**: Decrease threshold (1.5) or K (3-4)
- **Unbalanced Network**: Adjust both parameters incrementally

### 2. Data Quality

**Required Fields**:
- All numeric features must be present
- `segment_code` and `ecosystem_role` for categorical encoding
- No missing values in numeric features

**Data Preparation**:
- Handle missing values before upload
- Ensure consistent data types
- Validate feature ranges (no extreme outliers)

### 3. Relationship Interpretation

**Similarity Scores**:
- **>0.5**: Very strong relationship (rare, investigate)
- **0.3-0.5**: Strong relationship (high confidence)
- **0.2-0.3**: Moderate relationship (review context)
- **<0.2**: Weak relationship (may be noise)

**Context Matters**:
- Consider ecosystem role when interpreting relationships
- Check anchor group alignment
- Review transaction patterns for validation

### 4. Network Analysis

**Community Detection**:
- Use Louvain for graph-based communities
- Use DBSCAN for density-based clusters
- Compare results for validation

**Hub Accounts**:
- High degree centrality = many connections
- High betweenness = bridge between communities
- Both indicate influential accounts

### 5. Export and Analysis

**Relationship Export**:
- Export relationships CSV for external analysis
- Use in network analysis tools (Gephi, Cytoscape)
- Integrate with CRM systems

**Opportunity Export**:
- Export opportunity scores for prioritization
- Use for RM pipeline management
- Track relationship changes over time

## Technical Implementation

### Data Flow

```
1. Data Load (Demo/Upload)
   ↓
2. Feature Extraction (Numeric + Categorical)
   ↓
3. Preprocessing Pipeline
   - StandardScaler (numeric)
   - OneHotEncoder (categorical)
   ↓
4. KNN Model Training
   - NearestNeighbors.fit()
   ↓
5. Similarity Search
   - kneighbors() for each account
   ↓
6. Edge Creation
   - Filter by distance threshold
   - Calculate similarity scores
   - Add anchor metadata
   ↓
7. Graph Construction
   - NetworkX graph
   - Nodes = accounts
   - Edges = relationships
   ↓
8. Community Detection
   - Louvain or DBSCAN
   ↓
9. Visualization & Analysis
```

### Key Functions

#### `build_knn_pipeline(df, n_neighbors=6)`
- **Purpose**: Build preprocessing and KNN pipeline
- **Returns**: Trained pipeline, categorical features list
- **Location**: `app.py`

#### `find_neighbors_and_edges(df, pipeline, cat_features, max_distance=2.0)`
- **Purpose**: Find similar accounts and create relationship edges
- **Returns**: DataFrame with relationship edges
- **Location**: `app.py`

#### `build_graph(df, edges_df)`
- **Purpose**: Construct NetworkX graph from edges
- **Returns**: NetworkX Graph object
- **Location**: `app.py`

#### `get_account_neighbors(G, account_id, df)`
- **Purpose**: Get all neighbors for a specific account
- **Returns**: DataFrame with neighbor details
- **Location**: `app.py`

## Performance Considerations

### Dataset Size
- **<100 accounts**: Fast processing (<1 second)
- **100-1000 accounts**: Moderate processing (1-5 seconds)
- **1000-10000 accounts**: Slower processing (5-30 seconds)
- **>10000 accounts**: Consider batch processing

### Optimization Tips
- Reduce K neighbors for large datasets
- Increase similarity threshold to reduce edges
- Use DBSCAN instead of Louvain for very large networks
- Cache results using Streamlit's caching

## Troubleshooting

### Issue: No Relationships Found

**Possible Causes**:
1. Similarity threshold too strict
2. K neighbors too low
3. Data quality issues (missing features)
4. Accounts too dissimilar

**Solutions**:
- Increase similarity threshold (2.5-3.0)
- Increase K neighbors (8-10)
- Check data quality
- Verify feature ranges

### Issue: Too Many Weak Relationships

**Possible Causes**:
1. Similarity threshold too lenient
2. K neighbors too high
3. Data not properly normalized

**Solutions**:
- Decrease similarity threshold (1.5)
- Decrease K neighbors (3-4)
- Verify preprocessing pipeline

### Issue: Unexpected Relationships

**Possible Causes**:
1. Feature engineering issues
2. Data quality problems
3. Parameter misconfiguration

**Solutions**:
- Review feature values
- Check data quality
- Adjust parameters
- Validate with domain experts

## Future Enhancements

Planned improvements:
- **Temporal Analysis**: Track relationship changes over time
- **Weighted Features**: Allow feature importance weighting
- **Advanced Similarity Metrics**: Cosine similarity, Manhattan distance
- **Relationship Strength Categories**: Strong/Moderate/Weak classification
- **Automated Insights**: AI-generated relationship explanations
- **Real-time Updates**: Incremental relationship updates
- **Relationship Scoring**: Composite relationship quality score

## References

### Code References
- KNN Pipeline: `build_knn_pipeline()` function in `app.py`
- Relationship Detection: `find_neighbors_and_edges()` function in `app.py`
- Graph Construction: `build_graph()` function in `app.py`
- Neighbor Lookup: `get_account_neighbors()` function in `app.py`

### Related Documentation
- **[System Architecture Guide](ARCHITECTURE.md)**: Complete system architecture, data flow, and model details
- **[Lead & Opportunity Identification Guide](LEAD_OPPORTUNITY_IDENTIFICATION.md)**: How to identify quality leads and opportunities
- **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)**: Cross-anchor relationship detection
- **[Data Privacy & Anonymization Guide](DATA_PRIVACY_ANONYMIZATION.md)**: Data privacy compliance
- **[Main Documentation](README.md)**: Overall dashboard documentation

### External Resources
- [scikit-learn NearestNeighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [NetworkX Documentation](https://networkx.org/)
- [KNN Algorithm Explained](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

**Last Updated**: 2025-01-08  
**Version**: 1.0  
**Author**: KNN Dashboard Development Team


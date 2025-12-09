# Cross-Anchor Relationship Detection Documentation

## Overview

The KNN Dashboard now includes comprehensive functionality to identify and analyze relationships between accounts across different anchor groups. This feature enables Relationship Managers (RMs) to discover strategic connections, market expansion opportunities, and unusual transaction patterns that span multiple anchor ecosystems.

## What are Cross-Anchor Relationships?

Cross-anchor relationships are connections between accounts that belong to **different anchor groups** (e.g., Japfa_Group ↔ CP_Group). These relationships are particularly valuable because they:

- Reveal ecosystem bridges and strategic partnerships
- Identify market expansion opportunities
- Highlight potential NTB acquisition targets
- Detect unusual transaction patterns across anchor boundaries
- Support risk management and compliance monitoring

## Key Parameters

### 1. Edge-Level Parameters

Each relationship edge in the network graph now includes the following cross-anchor identification parameters:

#### `is_cross_anchor` (Boolean)
- **Purpose**: Primary flag indicating if a relationship connects accounts from different anchor groups
- **Logic**: `True` when:
  - Source and destination accounts belong to different anchor groups
  - Neither account is "Independent"
  - Both accounts have valid anchor group assignments
- **Usage**: Filters and highlights cross-anchor relationships in visualizations

#### `src_anchor` / `dst_anchor` (String)
- **Purpose**: Anchor group identifiers for source and destination accounts
- **Values**: "Japfa_Group", "CP_Group", "Charoen_Group", "Malindo_Group", "Independent"
- **Usage**: Tracks which anchor groups are connected through each relationship

#### `is_anchor_bridge` (Boolean)
- **Purpose**: Identifies relationships involving anchor-level accounts (high strategic value)
- **Logic**: `True` when either source or destination account has `anchor_level <= 1`
- **Usage**: Prioritizes relationships involving direct anchor connections

#### `anchor_distance` / `anchor_level_diff` (Integer)
- **Purpose**: Measures the difference in anchor hierarchy levels between connected accounts
- **Range**: 0-2 (0 = same level, 1 = one level apart, 2 = two levels apart)
- **Usage**: Quantifies proximity in the anchor hierarchy

### 2. Account-Level Metrics

#### Bridge Accounts
Accounts that connect multiple anchor groups are identified with the following metrics:

- **`cross_anchor_connections`**: Number of relationships to accounts in different anchor groups
- **`connected_anchors`**: Count of distinct anchor groups this account connects to
- **`avg_similarity`**: Average similarity score of cross-anchor relationships
- **`bridge_score`**: Composite score calculated as:
  ```
  bridge_score = (cross_anchor_connections × 0.4) + 
                 (connected_anchors × 0.3) + 
                 (avg_similarity × 30)
  ```

### 3. Opportunity Scoring Parameters

Cross-anchor opportunities are scored using:

- **`similarity`**: KNN-based behavioral similarity (0-1 scale)
- **`opportunity_score`**: Composite business value score (0-100) calculated as:
  ```
  base_score = similarity × 50
  + NTB_bonus (20 if either account is NTB)
  + lead_score_bonus (avg_lead_score × 0.3)
  + anchor_proximity_bonus (15 if anchor_level_diff <= 1)
  ```

## Dashboard Features

### Sidebar Configuration

#### Enable Cross-Anchor Relationship Detection
- **Location**: Sidebar → Cross-Anchor Analysis section
- **Default**: Enabled (checked)
- **Purpose**: Toggles calculation of cross-anchor metrics
- **Impact**: When disabled, cross-anchor analysis is skipped for performance

#### Highlight Cross-Anchor Edges in Network
- **Location**: Sidebar → Cross-Anchor Analysis section
- **Default**: Enabled (checked)
- **Purpose**: Visual distinction of cross-anchor relationships in network graphs
- **Visual Effect**: Cross-anchor edges appear in red (thicker, more visible) vs. gray for regular edges

### Ecosystem Overview Tab

#### Cross-Anchor Links Metric
- **Location**: Top metrics row (6th column)
- **Displays**: 
  - Count of cross-anchor relationships
  - Delta showing cross-anchor ratio as percentage
- **Example**: "15 (18.5%)" means 15 cross-anchor links out of 81 total relationships

### Anchor Analysis Tab

#### Cross-Anchor Relationship Analysis Section

**Metrics Dashboard**
- Total Relationships: All detected relationships
- Cross-Anchor Links: Count of cross-anchor relationships
- Cross-Anchor Ratio: Percentage of relationships that are cross-anchor
- Bridge Accounts: Number of accounts connecting multiple anchor groups

**Anchor Pair Connectivity Matrix**
- **Table View**: Shows relationships between each anchor pair
  - Anchor Pair: e.g., "Japfa_Group ↔ CP_Group"
  - Relationships: Count of connections
  - Avg Similarity: Average behavioral similarity
  - Strength: Categorized as Strong (>0.5), Moderate (0.3-0.5), or Weak (<0.3)
- **Visualizations**:
  - Bar chart: Relationship count by anchor pair
  - Scatter plot: Relationship strength vs. count

**Top Bridge Accounts**
- **Purpose**: Identifies accounts that connect different anchor ecosystems
- **Columns**:
  - Account ID and Legal Name (anonymized for UU PDP compliance)
  - Anchor Group
  - Ecosystem Role
  - Bridge Score (higher = more valuable connector)
  - Cross-Anchor Links count
  - Connected Anchors count
  - Average Similarity
  - BRI Status
- **Note**: All legal names are automatically anonymized using NER-based detection. See [Data Privacy & Anonymization Guide](DATA_PRIVACY_ANONYMIZATION.md) for details.

**Cross-Anchor Opportunities**
- **Purpose**: High-value relationships for RM action
- **Columns**:
  - Account 1 & 2 details (name, anchor, status) - names are anonymized
  - Similarity score
  - Opportunity Score (0-100)
  - Anchor Pair identifier
- **Sorting**: By opportunity score (highest first)
- **Note**: All account names displayed are anonymized for data privacy compliance

### Account Details Tab

#### Cross-Anchor Relationships Section
- **Location**: Below "Account in Network" visualization
- **Displays** (when account has cross-anchor connections):
  - Summary count of cross-anchor relationships
  - Detailed table of cross-anchor neighbors showing:
    - Account ID and Legal Name
    - Anchor Group
    - Relationship type (e.g., "Japfa_Group ↔ CP_Group")
    - Similarity score

#### Enhanced Neighbor List
- **New Column**: "Anchor Relationship"
- **Values**: 
  - "Same Anchor" for intra-anchor relationships
  - "Anchor1 ↔ Anchor2" for cross-anchor relationships
- **Visual Distinction**: Cross-anchor relationships are highlighted
- **Data Privacy**: All legal names are anonymized (see [Data Privacy & Anonymization Guide](DATA_PRIVACY_ANONYMIZATION.md))

## Use Cases

### 1. Market Expansion
**Scenario**: Identify accounts that bridge different anchor ecosystems for potential market entry.

**Parameters to Use**:
- `bridge_score` > 50
- `connected_anchors` >= 2
- `opportunity_score` > 60

**Action**: Target bridge accounts for relationship building and product cross-sell.

### 2. NTB Acquisition
**Scenario**: Find high-value NTB accounts connected to existing anchor relationships.

**Parameters to Use**:
- `is_cross_anchor` = True
- `opportunity_score` > 70
- At least one account has `bri_status` = "NTB"

**Action**: Prioritize these relationships for RM outreach and acquisition campaigns.

### 3. Risk Management
**Scenario**: Detect unusual cross-anchor transaction patterns that may indicate fraud or compliance issues.

**Parameters to Use**:
- `is_cross_anchor` = True
- `similarity` < 0.3 (unusual relationships)
- `anchor_level_diff` = 2 (distant hierarchy levels)

**Action**: Flag for compliance review and enhanced monitoring.

### 4. Strategic Partnerships
**Scenario**: Identify potential collaboration opportunities between anchor groups.

**Parameters to Use**:
- `anchor_pairs` with high relationship count (>10)
- `avg_similarity` > 0.5
- `is_anchor_bridge` = True

**Action**: Explore partnership opportunities and ecosystem synergies.

## Technical Implementation

### Data Flow

1. **Edge Detection** (`find_neighbors_and_edges()`)
   - KNN algorithm identifies similar accounts
   - For each relationship, anchor information is extracted
   - Cross-anchor flags and metrics are calculated

2. **Graph Construction** (`build_graph()`)
   - NetworkX graph includes cross-anchor attributes
   - Edge attributes stored for visualization and analysis

3. **Metrics Calculation** (`calculate_cross_anchor_metrics()`)
   - Aggregates cross-anchor statistics
   - Identifies bridge accounts
   - Builds anchor pair connectivity matrix

4. **Opportunity Identification** (`identify_cross_anchor_opportunities()`)
   - Scores each cross-anchor relationship
   - Ranks by business value
   - Filters for actionable opportunities

### Performance Considerations

- Cross-anchor analysis adds minimal overhead (<5% processing time)
- Metrics are calculated once per dashboard load
- Visual highlighting has no performance impact
- Can be disabled via sidebar checkbox for large datasets (>10,000 accounts)

## Data Requirements

### Required Columns in Input Data

For cross-anchor analysis to work, your dataset must include:

- **`anchor_group`**: Anchor group identifier (required)
- **`anchor_level`**: Hierarchy level (0=anchor, 1=direct, 2=indirect) (optional, defaults to 2)
- **`account_id`**: Unique account identifier (required)
- Standard KNN features (avg_txn_amount_30d, txn_count_30d, etc.)

### Default Values

If `anchor_group` is missing:
- Defaults to "Independent"
- Cross-anchor relationships will not be detected for these accounts

If `anchor_level` is missing:
- Defaults to 2 (indirect relationship)
- Bridge detection may be less accurate

## Best Practices

### 1. Similarity Threshold Tuning
- **Lower threshold** (1.5-2.0): More relationships, including weaker cross-anchor links
- **Higher threshold** (2.5-3.0): Fewer, stronger relationships, better signal-to-noise ratio
- **Recommendation**: Start with 2.0, adjust based on relationship count

### 2. Bridge Account Analysis
- Focus on accounts with `bridge_score` > 40
- Review accounts connecting 3+ anchor groups (high strategic value)
- Monitor bridge accounts for relationship changes over time

### 3. Opportunity Prioritization
- **High Priority**: `opportunity_score` > 70, at least one NTB account
- **Medium Priority**: `opportunity_score` 50-70, strong similarity (>0.5)
- **Low Priority**: `opportunity_score` < 50, but still cross-anchor

### 4. Anchor Pair Monitoring
- Track relationship counts between anchor pairs over time
- Alert on sudden increases (potential market shifts)
- Monitor similarity trends (relationship strength changes)

## Export and Integration

### CSV Export
Cross-anchor relationship data is included in:
- **Relationships Export**: Includes all edge parameters (`is_cross_anchor`, `src_anchor`, `dst_anchor`, etc.)
- **Opportunities Export**: Includes cross-anchor opportunity scores and anchor pair information

### API Integration (Future)
Parameters are structured for easy API integration:
```python
{
    "is_cross_anchor": bool,
    "src_anchor": str,
    "dst_anchor": str,
    "bridge_score": float,
    "opportunity_score": float
}
```

## Troubleshooting

### No Cross-Anchor Relationships Detected

**Possible Causes**:
1. Accounts belong to same anchor groups
2. Similarity threshold too strict
3. Missing `anchor_group` data
4. All accounts marked as "Independent"

**Solutions**:
- Verify `anchor_group` column has multiple distinct values
- Lower similarity threshold in sidebar
- Check data quality for anchor assignments
- Review KNN parameters (increase `n_neighbors`)

### Bridge Accounts Not Appearing

**Possible Causes**:
1. No accounts connect multiple anchor groups
2. Bridge score threshold too high
3. Insufficient relationships detected

**Solutions**:
- Verify cross-anchor relationships exist (check metrics)
- Review bridge score calculation
- Adjust similarity threshold to find more relationships

### Performance Issues

**Possible Causes**:
1. Large dataset (>10,000 accounts)
2. Very low similarity threshold (many relationships)

**Solutions**:
- Disable cross-anchor analysis temporarily
- Increase similarity threshold
- Reduce `n_neighbors` parameter
- Process in batches for very large datasets

## Future Enhancements

Planned improvements:
- Temporal analysis (track cross-anchor relationship changes over time)
- Predictive modeling (forecast cross-anchor opportunities)
- Automated alerts for high-value bridge accounts
- Integration with CRM systems for opportunity tracking
- Advanced visualization (sankey diagrams for anchor flows)

## Data Privacy Note

All account names and legal names displayed in cross-anchor analysis are automatically anonymized using NER-based detection to comply with UU PDP (Indonesian Personal Data Protection Law) and banking confidentiality requirements. The anonymization maintains consistent identifiers throughout the session while protecting sensitive customer information.

For detailed information on anonymization, see: **[Data Privacy & Anonymization Guide](DATA_PRIVACY_ANONYMIZATION.md)**

## Related Documentation

For system architecture and technical details:
- **[System Architecture Guide](ARCHITECTURE.md)**: Complete system architecture, data flow, model architecture, training and inference flows

For general ecosystem relationship identification (how KNN finds relationships):
- **[Ecosystem Relationship Identification Guide](ECOSYSTEM_RELATIONSHIP_IDENTIFICATION.md)**: Complete guide on KNN-based relationship detection, parameter tuning, and use cases

For identifying quality leads and opportunities:
- **[Lead & Opportunity Identification Guide](LEAD_OPPORTUNITY_IDENTIFICATION.md)**: Comprehensive guide on identifying quality leads, interpreting opportunity scores, and prioritizing accounts

## References

- Main Dashboard: `app.py`
- KNN Implementation: `find_neighbors_and_edges()` function
- Metrics Calculation: `calculate_cross_anchor_metrics()` function
- Opportunity Scoring: `identify_cross_anchor_opportunities()` function
- Anonymization: `anonymize_dataframe()` and related functions

---

**Last Updated**: 2025-01-08  
**Version**: 1.0  
**Author**: KNN Dashboard Development Team


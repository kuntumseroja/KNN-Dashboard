# Lead & Opportunity Identification Guide

## Overview

The OBS Account Relationship Dashboard uses a comprehensive **Opportunity Scoring System** to identify quality leads and prioritize Relationship Manager (RM) actions. This guide explains how to identify high-quality leads, interpret opportunity scores, and prioritize accounts for banking opportunities.

## What is a Quality Lead?

A **quality lead** is an account that demonstrates:
- **High Potential Value**: Strong financial indicators and growth potential
- **Network Influence**: Connections to other valuable accounts in the ecosystem
- **Strategic Position**: Important role in supply chain or anchor ecosystem
- **Acquisition Potential**: NTB (New-to-Bank) status or expansion opportunity
- **Behavioral Fit**: Transaction patterns similar to successful existing customers

## Opportunity Scoring System

### Composite Score (0-100)

The opportunity score is calculated from **5 component scores**:

```
Total Opportunity Score = 
  Network Score (20 points) +
  Influence Score (15 points) +
  Ecosystem Score (15 points) +
  Lead Score (25 points) +
  Potential Score (25 points)
```

### Component Breakdown

#### 1. Network Score (20 points)

**Purpose**: Measures account connectivity in the ecosystem network

**Calculation**:
```
Network Score = Degree Centrality × 20
```

**What it means**:
- **Degree Centrality**: Number of connections (relationships) an account has
- **Higher connections** = More influential in ecosystem
- **Range**: 0-20 points

**Interpretation**:
- **15-20 points**: Hub account (many connections, high influence)
- **10-15 points**: Well-connected account
- **5-10 points**: Moderate connections
- **0-5 points**: Isolated account (few connections)

**Why it matters**:
- Hub accounts can influence other accounts
- Well-connected accounts indicate ecosystem importance
- Network position suggests business relationships

#### 2. Influence Score (15 points)

**Purpose**: Measures how often an account bridges different communities

**Calculation**:
```
Influence Score = Betweenness Centrality × 15
```

**What it means**:
- **Betweenness Centrality**: How often account appears on shortest paths between other accounts
- **High betweenness** = Account is a bridge/connector
- **Range**: 0-15 points

**Interpretation**:
- **12-15 points**: Critical bridge account (connects communities)
- **8-12 points**: Important connector
- **4-8 points**: Moderate bridge role
- **0-4 points**: Not a bridge account

**Why it matters**:
- Bridge accounts connect different parts of ecosystem
- Strategic position for cross-selling
- Potential to influence multiple communities

#### 3. Ecosystem Score (15 points)

**Purpose**: Measures the size and strength of account's ecosystem community

**Calculation**:
```
Ecosystem Score = min(Community Size / 8, 1) × 15
```

**What it means**:
- **Community Size**: Number of accounts in the same community/cluster
- **Larger communities** = Stronger ecosystem position
- **Capped at 8 accounts** for scoring (communities >8 get full 15 points)
- **Range**: 0-15 points

**Interpretation**:
- **15 points**: Large community (8+ accounts)
- **10-15 points**: Medium-large community (5-8 accounts)
- **5-10 points**: Small-medium community (3-5 accounts)
- **0-5 points**: Small/isolated community (1-3 accounts)

**Why it matters**:
- Large communities indicate strong ecosystem relationships
- Community membership suggests business network
- Ecosystem position affects growth potential

#### 4. Lead Score (25 points)

**Purpose**: BRI lead scoring based on financial behavior and characteristics

**Calculation**:
```
Lead Score = (lead_score_bri / 100) × 25
```

**What it means**:
- **lead_score_bri**: Pre-calculated lead score (0-100)
- Based on transaction patterns, financial metrics, and business characteristics
- **Range**: 0-25 points

**Interpretation**:
- **20-25 points**: Excellent lead (lead_score_bri: 80-100)
- **15-20 points**: Good lead (lead_score_bri: 60-80)
- **10-15 points**: Average lead (lead_score_bri: 40-60)
- **0-10 points**: Low-quality lead (lead_score_bri: 0-40)

**Why it matters**:
- Direct indicator of account quality
- Based on comprehensive financial analysis
- Strong predictor of banking opportunity

#### 5. Potential Score (25 points)

**Purpose**: Measures acquisition and expansion potential

**Calculation**:
```
Potential Score = NTB Bonus + Anchor Proximity Bonus

NTB Bonus:
  +15 points if bri_status == "NTB"

Anchor Proximity Bonus:
  +10 points if anchor_level <= 1 (direct anchor relationship)
```

**What it means**:
- **NTB Status**: New-to-Bank accounts (not yet BRI customers)
- **Anchor Level**: Distance from anchor (0=anchor, 1=direct, 2=indirect)
- **Range**: 0-25 points

**Scoring Scenarios**:
- **25 points**: NTB + Direct anchor (15 + 10)
- **15 points**: NTB only
- **10 points**: Direct anchor only (existing customer)
- **0 points**: Existing customer + Indirect anchor

**Why it matters**:
- NTB accounts represent new revenue opportunities
- Direct anchor relationships indicate high-value accounts
- Potential score prioritizes acquisition targets

### Priority Classification

Based on total opportunity score:

| Score Range | Priority | Description | Action Required |
|-------------|----------|-------------|----------------|
| **≥70** | **High** | Top-quality leads | Immediate RM action |
| **45-69** | **Medium** | Good opportunities | RM follow-up within week |
| **<45** | **Low** | Standard accounts | Routine monitoring |

## Lead Quality Indicators

### High-Quality Lead Profile

A high-quality lead typically has:

1. **High Opportunity Score** (≥70)
   - Strong composite score across all components

2. **NTB Status** (bri_status = "NTB")
   - New-to-Bank = acquisition opportunity
   - Potential for new revenue

3. **High Lead Score** (lead_score_bri ≥ 75)
   - Strong financial indicators
   - Good transaction patterns

4. **Network Position**
   - High degree centrality (many connections)
   - High betweenness (bridge role)
   - Large community membership

5. **Anchor Proximity** (anchor_level ≤ 1)
   - Direct relationship to anchor
   - Strategic ecosystem position

6. **Strong Financial Metrics**
   - High turnover (turnover_90d)
   - Good transaction volume (txn_count_30d)
   - Healthy transaction amounts (avg_txn_amount_30d)

### Lead Quality Matrix

```
                    High Lead Score          Low Lead Score
                    (lead_score_bri ≥75)    (lead_score_bri <75)
                  
High Opportunity    ⭐⭐⭐ Premium Lead    ⭐⭐ Good Lead
(Score ≥70)         Immediate Action        Priority Action
                    
Medium Opportunity  ⭐⭐ Good Lead         ⭐ Standard Lead
(Score 45-69)       Follow-up              Monitor
                    
Low Opportunity     ⭐ Standard Lead       ⚠️ Low Priority
(Score <45)         Routine                Low Value
```

## Identifying Quality Leads in Dashboard

### RM Opportunities Tab

#### High Priority Accounts Section

**Location**: RM Opportunities Tab → "High Priority Accounts"

**What to Look For**:
- **Opportunity Score ≥70**: Top priority
- **NTB Status**: Acquisition targets
- **High Lead Score**: Quality indicators
- **Ecosystem Role**: Strategic positions

**Action Items**:
1. Review account details
2. Check connected accounts
3. Analyze opportunity score breakdown
4. Prioritize for RM outreach

#### NTB Acquisition Targets Section

**Location**: RM Opportunities Tab → "NTB Acquisition Targets"

**What to Look For**:
- **NTB Status**: New-to-Bank accounts
- **High Lead Score** (≥75): Quality indicators
- **High Turnover**: Revenue potential
- **Ecosystem Role**: Strategic positions

**Action Items**:
1. Sort by lead_score_bri (highest first)
2. Review turnover and transaction patterns
3. Check ecosystem relationships
4. Prioritize for acquisition campaigns

### Account Details Tab

#### Opportunity Score Breakdown

**Location**: Account Details Tab → "Opportunity Score" section

**What to Review**:
- **Total Score**: Overall opportunity (0-100)
- **Network Score**: Connection strength
- **Influence Score**: Bridge importance
- **Lead Score**: Financial quality
- **Potential Score**: Acquisition potential

**Interpretation**:
- **High Total + High Lead**: Excellent quality lead
- **High Total + High Potential**: Strong acquisition target
- **High Network + High Influence**: Ecosystem hub (influence others)

#### Connected Accounts Analysis

**Location**: Account Details Tab → "Connected Accounts"

**What to Look For**:
- **High Similarity Scores** (>0.3): Strong relationships
- **NTB Accounts**: Connected acquisition targets
- **High-Value Accounts**: Connected to valuable customers
- **Cross-Anchor Relationships**: Strategic connections

**Action Items**:
1. Identify similar NTB accounts
2. Find accounts connected to high-value customers
3. Discover relationship patterns
4. Plan cross-sell strategies

### Ecosystem Overview Tab

#### Top Connected Accounts (Ecosystem Hubs)

**Location**: Ecosystem Overview Tab → "Top Connected Accounts"

**What to Look For**:
- **High Connection Count**: Hub accounts
- **High Lead Score**: Quality indicators
- **NTB Status**: Acquisition opportunities
- **Strategic Roles**: Anchor_Corporate, Feed_Mill, etc.

**Why It Matters**:
- Hub accounts influence others
- Acquiring hubs can unlock network effects
- Strategic positions in ecosystem

## Use Cases

### Use Case 1: NTB Acquisition Targeting

**Goal**: Identify high-quality NTB accounts for acquisition

**Steps**:
1. Go to **RM Opportunities Tab**
2. Navigate to **NTB Acquisition Targets** section
3. Sort by **Lead Score** (descending)
4. Filter for **Opportunity Score ≥70** (if available)
5. Review top accounts:
   - High lead_score_bri (≥75)
   - High turnover_90d
   - Strategic ecosystem_role
   - Good transaction patterns

**Quality Indicators**:
- ✅ NTB status
- ✅ Lead score ≥75
- ✅ Opportunity score ≥70
- ✅ High turnover
- ✅ Strategic ecosystem role

**Action**: Prioritize for RM outreach and acquisition campaigns

### Use Case 2: High-Value Lead Identification

**Goal**: Find accounts with highest opportunity scores

**Steps**:
1. Go to **RM Opportunities Tab**
2. Navigate to **High Priority Accounts** section
3. Review accounts sorted by **Opportunity Score**
4. Analyze score breakdown:
   - Network Score (connectivity)
   - Influence Score (bridge role)
   - Lead Score (financial quality)
   - Potential Score (acquisition potential)

**Quality Indicators**:
- ✅ Opportunity score ≥70
- ✅ High lead score (≥20 points)
- ✅ High network score (≥15 points)
- ✅ NTB status or direct anchor relationship

**Action**: Immediate RM action for top-scoring accounts

### Use Case 3: Ecosystem Hub Targeting

**Goal**: Identify influential accounts that can unlock network effects

**Steps**:
1. Go to **Ecosystem Overview Tab**
2. Review **Top Connected Accounts** section
3. Identify accounts with:
   - High connection count
   - High lead score
   - NTB status (if available)
4. Check **Account Details** for:
   - High influence score
   - Large community size
   - Bridge role indicators

**Quality Indicators**:
- ✅ High degree centrality (many connections)
- ✅ High betweenness centrality (bridge role)
- ✅ Large community membership
- ✅ Strategic ecosystem role

**Action**: Target hub accounts for acquisition (unlocks network effects)

### Use Case 4: Similar Account Discovery

**Goal**: Find accounts similar to high-value existing customers

**Steps**:
1. Go to **Account Details Tab**
2. Select a high-value existing customer
3. Review **Connected Accounts** section
4. Filter for:
   - High similarity scores (>0.3)
   - NTB status
   - High lead scores
5. Analyze relationship patterns

**Quality Indicators**:
- ✅ High similarity (>0.3) to successful customer
   - ✅ NTB status
   - ✅ Similar transaction patterns
   - ✅ Comparable lead scores

**Action**: Target similar NTB accounts for cross-sell opportunities

### Use Case 5: Cross-Anchor Opportunity Identification

**Goal**: Find high-value relationships across anchor groups

**Steps**:
1. Go to **Anchor Analysis Tab**
2. Navigate to **Cross-Anchor Opportunities** section
3. Review opportunities sorted by **Opportunity Score**
4. Analyze:
   - Similarity scores
   - NTB status
   - Anchor pair relationships
   - Lead scores

**Quality Indicators**:
- ✅ High opportunity score (≥70)
- ✅ High similarity (>0.5)
- ✅ At least one NTB account
- ✅ Direct anchor relationships (anchor_level ≤1)

**Action**: Prioritize cross-anchor opportunities for strategic partnerships

## Best Practices

### 1. Multi-Factor Analysis

**Don't rely on single metric**:
- ✅ Use opportunity score (composite)
- ✅ Review individual components
- ✅ Consider ecosystem context
- ✅ Check relationship patterns

**Example**:
- Account A: High lead score but low network
- Account B: Medium lead score but high network + NTB
- **Account B may be better** (network effects + acquisition)

### 2. Prioritization Framework

**Priority Order**:
1. **High Priority** (Score ≥70) + NTB
2. **High Priority** (Score ≥70) + Existing
3. **Medium Priority** (Score 45-69) + NTB
4. **Medium Priority** (Score 45-69) + Existing
5. **Low Priority** (Score <45) - Monitor only

### 3. Context Matters

**Consider**:
- **Ecosystem Role**: Strategic positions (Anchor_Corporate, Feed_Mill)
- **Anchor Proximity**: Direct relationships (anchor_level ≤1)
- **Network Position**: Hub accounts (high connections)
- **Community Size**: Large communities (ecosystem strength)

### 4. Relationship Analysis

**Review Connected Accounts**:
- Similar accounts may have similar potential
- Network effects can multiply value
- Cross-anchor relationships indicate strategic value
- Bridge accounts unlock multiple opportunities

### 5. Score Interpretation

**Opportunity Score Ranges**:
- **80-100**: Exceptional leads (rare, investigate)
- **70-79**: High-quality leads (top priority)
- **60-69**: Good opportunities (strong potential)
- **45-59**: Moderate opportunities (follow-up)
- **<45**: Standard accounts (routine monitoring)

### 6. Lead Score Validation

**Cross-Reference**:
- Opportunity score vs. Lead score
- Network metrics vs. Financial metrics
- Ecosystem position vs. Transaction patterns
- Relationship strength vs. Business value

## Cross-Anchor Opportunity Scoring

### Special Scoring for Cross-Anchor Relationships

Cross-anchor opportunities use a different scoring formula:

```
Cross-Anchor Opportunity Score =
  Base Score (similarity × 50) +
  NTB Bonus (+20 if either account is NTB) +
  Lead Score Bonus (avg_lead_score × 0.3) +
  Anchor Proximity Bonus (+15 if anchor_level_diff ≤ 1)
```

**Maximum Score**: 100 points

### Quality Indicators for Cross-Anchor Opportunities

**High-Quality Cross-Anchor Opportunity**:
- ✅ High similarity (>0.5)
- ✅ At least one NTB account
- ✅ High lead scores (both accounts)
- ✅ Direct anchor relationships (anchor_level ≤1)
- ✅ Opportunity score ≥70

**Why Cross-Anchor Matters**:
- Bridge different ecosystems
- Strategic partnership opportunities
- Market expansion potential
- Competitive intelligence

See **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)** for detailed documentation.

## Dashboard Features for Lead Identification

### RM Opportunities Tab

#### Priority Distribution
- **Visual**: Pie chart showing High/Medium/Low priority breakdown
- **Purpose**: Quick overview of opportunity pipeline
- **Action**: Focus on High priority segment

#### Opportunity Score Distribution
- **Visual**: Histogram showing score distribution
- **Purpose**: Understand overall lead quality
- **Action**: Identify score clusters

#### High Priority Accounts Table
- **Columns**: Account, Legal Name, Role, Anchor, BRI Status, Segment, Opp Score, Lead Score
- **Sorting**: By opportunity score (highest first)
- **Action**: Review and prioritize for RM action

#### NTB Acquisition Targets Table
- **Columns**: Account, Legal Name, Role, Anchor, Current Bank, Lead Score, Turnover
- **Sorting**: By lead score (highest first)
- **Action**: Prioritize for acquisition campaigns

### Account Details Tab

#### Opportunity Score Breakdown
- **Metrics**: Total Score, Network Score, Influence Score, Lead Score, Potential Score
- **Purpose**: Understand score composition
- **Action**: Identify improvement areas

#### Connected Accounts
- **Information**: Similar accounts with similarity scores
- **Purpose**: Find similar opportunities
- **Action**: Target similar NTB accounts

### Export Functionality

#### Opportunities Export
- **File**: `rm_opportunities.csv`
- **Columns**: account_id, total_score, network_score, influence_score, ecosystem_score, lead_score, potential_score, priority
- **Purpose**: External analysis and CRM integration
- **Action**: Import to CRM or analysis tools

## Troubleshooting

### Issue: Low Opportunity Scores

**Possible Causes**:
1. Accounts have few connections (low network score)
2. Accounts not in large communities (low ecosystem score)
3. Low lead scores (lead_score_bri)
4. Existing customers without anchor proximity (low potential score)

**Solutions**:
- Review individual score components
- Check if accounts are isolated (few relationships)
- Verify lead_score_bri values
- Consider NTB accounts for higher potential scores

### Issue: High Scores but Low Quality

**Possible Causes**:
1. Network effects inflating scores
2. Large communities but weak relationships
3. High connections but low lead scores

**Solutions**:
- Review lead score component (should be ≥20 points)
- Check relationship quality (similarity scores)
- Validate financial metrics (turnover, transactions)
- Cross-reference with business context

### Issue: Missing NTB Opportunities

**Possible Causes**:
1. Dataset has few NTB accounts
2. NTB accounts have low scores
3. Filter settings too strict

**Solutions**:
- Check NTB count in dataset
- Review NTB account scores individually
- Lower opportunity score threshold
- Focus on lead score for NTB accounts

## Performance Metrics

### Lead Quality Metrics

Track these metrics over time:

1. **High Priority Lead Count**: Accounts with score ≥70
2. **NTB High-Value Count**: NTB accounts with lead_score ≥75
3. **Average Opportunity Score**: Overall pipeline quality
4. **Conversion Rate**: Leads converted to customers
5. **Average Lead Score**: Overall lead quality

### Success Indicators

**Good Pipeline Health**:
- 10-20% of accounts are High Priority
- 5-10% are NTB High-Value
- Average opportunity score >50
- High priority accounts have lead_score ≥75

## Future Enhancements

Planned improvements:
- **Predictive Lead Scoring**: ML models for conversion prediction
- **Lead Lifecycle Tracking**: Track leads through acquisition process
- **Automated Prioritization**: AI-based lead ranking
- **Relationship-Based Scoring**: Boost scores based on connected account quality
- **Temporal Analysis**: Track lead score changes over time
- **Custom Scoring Models**: Configurable scoring weights
- **Lead Quality Categories**: Premium/Standard/Basic classification

## References

### Code References
- Opportunity Scoring: `calculate_opportunity_score()` function in `app.py`
- Cross-Anchor Opportunities: `identify_cross_anchor_opportunities()` function in `app.py`
- Priority Classification: Priority assignment logic in `calculate_opportunity_score()`

### Related Documentation
- **[System Architecture Guide](ARCHITECTURE.md)**: Complete system architecture and model details
- **[Ecosystem Relationship Identification Guide](ECOSYSTEM_RELATIONSHIP_IDENTIFICATION.md)**: How relationships are identified
- **[Cross-Anchor Relationships Guide](CROSS_ANCHOR_RELATIONSHIPS.md)**: Cross-anchor opportunity detection
- **[Main Documentation](README.md)**: Overall dashboard documentation

### External Resources
- [Network Centrality Measures](https://en.wikipedia.org/wiki/Centrality)
- [Lead Scoring Best Practices](https://www.salesforce.com/resources/articles/lead-scoring/)
- [Customer Acquisition Strategies](https://www.hubspot.com/customer-acquisition)

---

**Last Updated**: 2025-01-08  
**Version**: 1.0  
**Author**: KNN Dashboard Development Team


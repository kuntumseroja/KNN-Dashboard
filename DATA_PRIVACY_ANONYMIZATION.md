# Data Privacy & Anonymization Documentation

## Overview

The OBS Account Relationship Dashboard implements **automatic NER-based anonymization** to ensure compliance with:
- **UU PDP** (Undang-Undang Perlindungan Data Pribadi) - Indonesian Personal Data Protection Law
- **Banking Confidentiality** requirements (Kerahasiaan Customer)

All customer data is automatically anonymized before display, ensuring that sensitive information such as company names and person names are protected while maintaining data utility for analysis.

## How Anonymization Works

### Automatic Processing

1. **Data Ingestion**: When data is loaded (either from demo generation or file upload), the anonymization process runs automatically
2. **Entity Detection**: The system identifies sensitive entities using:
   - **Primary Method**: spaCy NER model (if installed) - provides high accuracy entity recognition
   - **Fallback Method**: Regex-based pattern matching for Indonesian business entities
3. **Anonymization**: Detected entities are replaced with consistent anonymized identifiers
4. **Session Persistence**: The same entity always receives the same anonymized identifier throughout the session

### Entity Types Detected

The system anonymizes the following entity types:

#### Organizations (ORG)
- **Indonesian Company Types**: PT, CV, UD, PD, Perum, Persero, Koperasi
- **Patterns Detected**:
  - `PT [Company Name]`
  - `CV [Company Name]`
  - `UD [Company Name]`
  - Other Indonesian business entity formats
- **Anonymization Format**: `PT ENTITY_[8-char-hash]` (preserves company type prefix)

#### Persons (PERSON)
- **Patterns**: Indonesian name patterns (2-4 capitalized words)
- **Anonymization Format**: `PERSON_[8-char-hash]`

### Anonymization Algorithm

```python
# Simplified pseudocode
1. Load text (e.g., legal_name field)
2. Detect entities using NER or regex
3. For each detected entity:
   - Generate consistent hash: MD5(entity)[:8].upper()
   - Create anonymized identifier based on entity type
   - Store mapping in session state
4. Replace all occurrences of entity with anonymized version
5. Return anonymized text
```

### Fields Anonymized

Currently, the following fields are automatically anonymized:
- **`legal_name`**: Company/business legal names

Additional fields can be added to the anonymization list as needed.

## Installation & Configuration

### Basic Setup (Regex-Based)

The dashboard works out-of-the-box with regex-based anonymization. No additional installation required.

### Enhanced Setup (NER-Based)

For improved accuracy, install spaCy with a language model:

```bash
# Install spaCy
pip install spacy

# Download a language model (choose one):
python -m spacy download id_core_web_sm    # Indonesian (recommended for Indonesian data)
python -m spacy download xx_ent_wiki_sm    # Multilingual (supports multiple languages)
python -m spacy download en_core_web_sm    # English (fallback option)
```

### Model Priority

The system tries to load models in this order:
1. `id_core_web_sm` (Indonesian) - Best for Indonesian business data
2. `xx_ent_wiki_sm` (Multilingual) - Good general-purpose option
3. `en_core_web_sm` (English) - Fallback option

If no model is available, the system automatically falls back to regex-based detection.

### Verification

Check anonymization status in the dashboard sidebar:
- **"✓ Using NER-based anonymization (spaCy)"** - NER model loaded successfully
- **"ℹ Using regex-based anonymization"** - Using fallback method

## Technical Implementation

### Core Functions

#### `load_ner_model()`
- **Purpose**: Loads spaCy NER model with fallback chain
- **Returns**: `(nlp_model, is_available)` tuple
- **Caching**: Results are cached using `@st.cache_resource` for performance

#### `detect_entities_regex(text)`
- **Purpose**: Regex-based entity detection for Indonesian patterns
- **Returns**: Dictionary with `PERSON`, `ORG`, `LOC` entity lists
- **Patterns**: Handles Indonesian company types and name patterns

#### `anonymize_text(text, anonymization_map, use_ner=True)`
- **Purpose**: Anonymizes a single text string
- **Parameters**:
  - `text`: Text to anonymize
  - `anonymization_map`: Dictionary for consistent entity mapping
  - `use_ner`: Whether to use NER (falls back to regex if unavailable)
- **Returns**: Anonymized text string

#### `anonymize_dataframe(df, anonymize_fields=None)`
- **Purpose**: Applies anonymization to entire dataframe
- **Parameters**:
  - `df`: Pandas DataFrame to anonymize
  - `anonymize_fields`: List of column names to anonymize (default: `["legal_name"]`)
- **Returns**: Anonymized DataFrame
- **Session State**: Maintains anonymization mapping in `st.session_state.anonymization_map`

### Data Flow

```
1. Data Load (Demo/Upload)
   ↓
2. anonymize_dataframe() called
   ↓
3. For each field in anonymize_fields:
   ↓
4. anonymize_text() called for each value
   ↓
5. Entity detection (NER or regex)
   ↓
6. Hash generation & mapping
   ↓
7. Text replacement
   ↓
8. Return anonymized dataframe
   ↓
9. All displays use anonymized data
```

## Compliance Features

### UU PDP Compliance

The anonymization system ensures compliance with Indonesian Personal Data Protection Law by:
- **Automatic Processing**: No manual intervention required
- **Consistent Anonymization**: Same entity = same anonymized identifier
- **Non-Reversible**: Hash-based identifiers prevent reverse engineering
- **Always Enabled**: Cannot be disabled to ensure compliance

### Banking Confidentiality

Complies with banking confidentiality requirements (Kerahasiaan Customer) by:
- **Pre-Display Processing**: Data anonymized before any UI rendering
- **Export Protection**: Exported data also contains anonymized values
- **Session Isolation**: Anonymization mapping is session-specific

## Usage Examples

### Example 1: Company Name Anonymization

**Input**:
```
legal_name: "PT Japfa Comfeed Indonesia Tbk"
```

**Output** (anonymized):
```
legal_name: "PT ENTITY_A3F2B9C1"
```

### Example 2: Person Name Anonymization

**Input**:
```
legal_name: "CV Sumber Ayam Sejahtera"
```

**Output** (anonymized):
```
legal_name: "CV ENTITY_D4E5F6A2"
```

### Example 3: Consistent Mapping

If the same entity appears multiple times:
- First occurrence: `"PT Japfa Comfeed Indonesia Tbk"` → `"PT ENTITY_A3F2B9C1"`
- Second occurrence: `"PT Japfa Comfeed Indonesia Tbk"` → `"PT ENTITY_A3F2B9C1"` (same hash)

## Best Practices

### 1. Model Selection
- **For Indonesian Data**: Use `id_core_web_sm` for best accuracy
- **For Mixed Languages**: Use `xx_ent_wiki_sm`
- **For English Only**: Use `en_core_web_sm`

### 2. Data Quality
- Ensure `legal_name` field contains properly formatted company/person names
- Avoid special characters that might interfere with entity detection
- Use standard Indonesian business entity prefixes (PT, CV, UD, etc.)

### 3. Performance
- NER models are cached after first load
- Regex-based fallback is fast and suitable for most use cases
- Large datasets (>10,000 rows) may take a few seconds to anonymize

### 4. Verification
- Check sidebar status to confirm anonymization method
- Review anonymized data in dashboard displays
- Verify exports contain anonymized values

## Troubleshooting

### Issue: Anonymization Not Working

**Symptoms**: Real names still visible in dashboard

**Solutions**:
1. Check sidebar status - ensure anonymization is enabled (always enabled by default)
2. Verify data is being processed through `anonymize_dataframe()`
3. Check browser console for errors
4. Refresh the page to reset session state

### Issue: Inconsistent Anonymization

**Symptoms**: Same entity has different anonymized identifiers

**Solutions**:
1. Clear session state: Refresh the page
2. Ensure same session is used throughout
3. Check that `anonymization_map` is properly maintained in session state

### Issue: Poor Entity Detection

**Symptoms**: Some entities not being anonymized

**Solutions**:
1. Install spaCy model for better accuracy
2. Check entity patterns match expected formats
3. Review regex patterns in `detect_entities_regex()` function
4. Verify entity names follow Indonesian business naming conventions

### Issue: Performance Problems

**Symptoms**: Slow dashboard loading

**Solutions**:
1. Use regex-based anonymization (faster, no model loading)
2. Reduce dataset size for testing
3. Check spaCy model size (smaller models load faster)
4. Consider batch processing for very large datasets

## Security Considerations

### Hash Algorithm
- Uses MD5 hash (first 8 characters)
- **Purpose**: Consistency, not security
- **Note**: MD5 is used for deterministic mapping, not cryptographic security

### Session State
- Anonymization mapping stored in Streamlit session state
- **Scope**: Session-specific (not persistent across sessions)
- **Access**: Only accessible within the same Streamlit session

### Data Storage
- Original data is not stored after anonymization
- Anonymized data is what's displayed and exported
- No reverse mapping is maintained

## Future Enhancements

Planned improvements:
- Support for additional entity types (locations, dates, etc.)
- Custom anonymization rules per field
- Anonymization audit logging
- Integration with external anonymization services
- Support for multiple languages beyond Indonesian
- Advanced NER models with fine-tuning capabilities

## References

### Code References
- Main anonymization functions: `app.py` (lines ~140-280)
- NER model loading: `load_ner_model()` function
- Regex detection: `detect_entities_regex()` function
- Dataframe processing: `anonymize_dataframe()` function

### External Resources
- [UU PDP (Indonesian Personal Data Protection Law)](https://jdih.kominfo.go.id/produk_hukum/view/id/1337/t/peraturan+menteri+komunikasi+dan+informatika+nomor+20+tahun+2016)
- [spaCy Documentation](https://spacy.io/)
- [Indonesian spaCy Model](https://spacy.io/models/id)

## Support

For questions or issues related to anonymization:
1. Check this documentation
2. Review dashboard sidebar status
3. Verify data format matches expected patterns
4. Check console for error messages

---

**Last Updated**: 2025-01-08  
**Version**: 1.0  
**Author**: KNN Dashboard Development Team


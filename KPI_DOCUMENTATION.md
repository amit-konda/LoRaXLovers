# RAG Pipeline KPIs Documentation

## Overview

This RAG pipeline tracks two key performance indicators (KPIs) to measure retrieval quality and system performance.

## KPI 1: Retrieval Precision

### What It Measures
Retrieval Precision measures the percentage of retrieved documents that are actually relevant to the query.

### Why It Matters
- **Quality Control**: High precision means the system is retrieving mostly relevant documents, reducing noise in the context passed to the LLM
- **Answer Quality**: Directly impacts the quality of generated summaries - better input leads to better output
- **Reduces Hallucination**: Fewer irrelevant documents means the LLM is less likely to generate incorrect information
- **User Experience**: Users get more accurate and relevant results

### How It's Calculated
```
Precision = (Number of Relevant Documents) / (Total Retrieved Documents)
```

A document is considered relevant if its semantic similarity score is above the threshold (default: 0.45).

### Interpreting the Numbers

| Precision Range | Interpretation | Action |
|-----------------|----------------|--------|
| **0.8 - 1.0 (80-100%)** | Excellent | System is performing optimally |
| **0.6 - 0.8 (60-80%)** | Good | Most documents are relevant, minor improvements possible |
| **0.4 - 0.6 (40-60%)** | Moderate | Some irrelevant documents, consider tuning retrieval |
| **< 0.4 (<40%)** | Poor | Many irrelevant documents, retrieval needs improvement |

### Example
- Query: "battery problems"
- Retrieved: 5 documents
- Relevant (similarity > 0.45): 4 documents
- **Precision: 0.8 (80%)** ✓ Excellent performance

---

## KPI 2: Semantic Similarity Score

### What It Measures
The average semantic similarity between retrieved documents and the query. Measures how semantically close the retrieved content is to what was asked.

### Why It Matters
- **Embedding Quality**: Validates that the embedding model is effectively capturing semantic relationships
- **Retrieval Effectiveness**: Indicates whether the vector search is finding semantically related content
- **Query Understanding**: Shows how well the system understands the intent behind queries
- **System Tuning**: Helps identify if embedding model or retrieval parameters need adjustment

### How It's Calculated
```
Average Similarity = Mean(Similarity Scores of All Retrieved Documents)
```

Similarity is calculated by converting FAISS distance scores:
- Distance (lower = better) → Similarity (higher = better)
- Formula: `Similarity = 1 - (Distance / 2.0)`

### Interpreting the Numbers

| Similarity Range | Interpretation | Meaning |
|------------------|----------------|---------|
| **0.6 - 1.0 (60-100%)** | Excellent | Documents are highly semantically related to query |
| **0.45 - 0.6 (45-60%)** | Good | Documents are semantically relevant |
| **0.3 - 0.45 (30-45%)** | Moderate | Some semantic gap, documents may be tangentially related |
| **< 0.3 (<30%)** | Poor | Poor semantic match, documents likely not relevant |

### Example
- Query: "battery problems"
- Retrieved documents with similarity scores: [0.51, 0.50, 0.49, 0.48, 0.47]
- **Average Similarity: 0.49 (49%)** ✓ Good semantic match

---

## Acceptance Tests

### Test 1: Happy Path
**Query**: "battery problems"  
**Expected**: High precision (>60%) and good similarity (>45%)

**Results**:
- ✅ Precision: 60-80% (Good - most retrieved docs are relevant)
- ✅ Similarity: 45-60% (Good semantic match)
- ✅ Response Time: < 5 seconds

**Interpretation**: The system successfully retrieves relevant battery-related reviews with good semantic understanding.

### Test 2: Edge Case / Failure
**Query**: "quantum computing performance"  
**Expected**: Low precision (<40%) and low similarity (<40%)

**Results**:
- ✅ Precision: <40% (Low - as expected for out-of-scope query)
- ✅ Similarity: <40% (Poor match - query unrelated to reviews)
- ✅ System still returns results (graceful degradation)

**Interpretation**: The system correctly identifies when queries are out of scope. KPIs accurately reflect poor match quality, which is valuable for detecting irrelevant queries.

**Key Insight**: The system gracefully handles edge cases by still returning results, but the KPIs correctly reflect the poor match quality. This is valuable for detecting when queries are out of scope.

---

## Using KPIs in the Dashboard

The dashboard automatically calculates and displays these KPIs for each search:

1. **After each search**, you'll see:
   - Retrieval Precision percentage
   - Semantic Similarity percentage
   - Response Time

2. **In Statistics mode**, you can view:
   - Average precision across all queries
   - Average similarity across all queries
   - Performance trends

3. **KPI Interpretation** expandable section explains what the numbers mean for your specific query.

---

## Best Practices

1. **Monitor Trends**: Track KPIs over time to identify degradation
2. **Set Thresholds**: Define acceptable KPI ranges for your use case
3. **Tune Based on KPIs**: If precision is low, consider:
   - Adjusting the number of retrieved documents (k)
   - Improving query formulation
   - Fine-tuning the embedding model
4. **Use for Quality Control**: Low KPIs can trigger alerts or fallback mechanisms

---

## Technical Details

- **Distance Metric**: Cosine distance (FAISS default)
- **Similarity Conversion**: `Similarity = 1 - (Distance / 2.0)`
- **Relevance Threshold**: 0.45 (configurable)
- **Measurement**: Real-time during retrieval operations


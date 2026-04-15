# Gene Extraction Build Process

## 0) Pipeline Overview

Input PDF
-> Text extraction and cleanup
-> Section and sentence structuring
-> High-recall candidate mining
-> Precision validation layers
-> Optional web validation
-> Consensus filtering
-> Ranked final output

Use strict filtering. Precision improves by rejecting weak candidates.

---

## 1) Input and Text Preparation

### 1.1 PDF acquisition
- Accept research paper PDF.
- Check whether text is selectable.
- If scanned image PDF, run OCR (example: Tesseract).

### 1.2 Text extraction
- Extract text from all pages.
- Preserve coarse structure when possible (headings, paragraph blocks).

### 1.3 Cleanup and normalization
- Remove non-content sections when detectable:
  - References
  - Acknowledgements
  - Author affiliations
- Remove citation patterns like `(Smith et al., 2020)` where possible.
- Normalize whitespace and remove noisy symbols.

Output gate for Stage 1:
- Clean scientific text is available.
- Major noise sections are removed.

---

## 2) Document Structuring

### 2.1 Section segmentation
Split into:
- Abstract
- Introduction
- Methods
- Results
- Discussion

Prioritize for extraction confidence:
1. Results
2. Discussion
3. Abstract

Lower priority:
- Methods
- Introduction

Always ignore for extraction:
- References
- Acknowledgements

### 2.2 Sentence segmentation
- Split each retained section into sentences.
- Keep sentence boundaries and section labels for downstream context scoring.

Output gate for Stage 2:
- Every sentence has section context.
- Candidate lookup can map token -> sentence(s).

---

## 3) High-Recall Candidate Mining

### 3.1 Candidate extraction (intentionally broad)
Extract gene-like tokens, including:
- Uppercase symbols: `TP53`, `EGFR`
- Alphanumeric forms: `BRCA1`
- Hyphenated forms when relevant

### 3.2 Frequency filtering
- Count candidate occurrences in the full retained text.
- Remove candidates appearing only once (configurable).

Note:
- This stage should over-capture; false positives are expected.

Output gate for Stage 3:
- Candidate set has enough recall for later precision filtering.

---

## 4) Precision Validation Core

### 4.1 Context detection
For each candidate:
- Collect all sentences containing the token.
- Check nearby biological context terms, for example:
  - gene
  - protein
  - expression
  - mutation
  - pathway
  - regulation

### 4.2 Context scoring
Compute a context score from:
- Biological keyword density in sentence/window.
- Quality of nearby action words and biological nouns.
- Section weight (Results/Discussion weighted higher).

### 4.3 Pattern validation
Validate form-based rules:
- Uppercase-dominant token
- Optional digit signal (strong but not mandatory)
- Length in reasonable range (for example 3-10)

Reject obvious noise:
- Very short tokens
- Common non-gene words

### 4.4 Ambiguity filtering
- Remove known ambiguous uppercase words (examples: `MAP`, `SET`, `CAT`) unless context score is very high.
- Remove acronyms with clear non-biological usage.

Output gate for Stage 4:
- Candidate list is high precision, not just high recall.

---

## 5) Optional External Validation (Web)

Use this only when higher precision is required and latency is acceptable.

### 5.1 Query construction
For each candidate, build queries like:
- `[TOKEN] gene protein function`

### 5.2 Multi-source checks
Validate against at least two independent source types:
- Wikipedia or equivalent encyclopedia pages
- Scientific snippets/pages
- Biological reference pages (scraped text if no API)

### 5.3 Content signal detection
From fetched content, score presence of terms such as:
- gene
- protein
- expression
- DNA

### 5.4 External confidence scoring
Increase confidence when:
- Biological terms are present
- Multiple sources agree

### 5.5 Caching
- Cache validation results by candidate token.
- Reuse cache to reduce repeated queries and rate-limit risk.

Output gate for Stage 5:
- Optional external confidence per candidate is available.

---

## 6) Consensus Filtering

Final candidate must pass multiple independent checks.

Recommended acceptance rule:
- Frequency threshold: pass
- Context validation: pass
- Pattern validation: pass
- Ambiguity filter: pass
- External validation: pass (if enabled)

Use a score threshold rather than hard yes/no where possible.

Example weighted score:
- Appears multiple times: +2
- Strong biological context: +3
- Gene-like pattern: +1
- External validation: +3
- Ambiguous usage: -3

Accept if total score >= configured threshold.

Output gate for Stage 6:
- Final accepted gene list with reasons/scores.

---

## 7) Post-Processing and Output

### 7.1 Deduplication and normalization
- Deduplicate tokens.
- Normalize format consistently.

### 7.2 Ranking
Rank by combined evidence:
- Frequency
- Context score
- External confidence (if used)

### 7.3 Output payload
Return:
- Final gene list
- Optional metadata per gene:
  - frequency
  - example sentence(s)
  - confidence score
  - validation flags per filter layer

---

## 8) Operational Rules for High Precision

- Prefer rejection over acceptance when uncertain.
- Require evidence across multiple sections when possible.
- Give higher weight to Results and Discussion.
- Use sentence-level behavior, not token shape alone.

---

## 9) Known Limitations

Without a curated gene database or specialized biomedical model:
- No guaranteed ground-truth validation
- Gene synonym resolution is weak (example: `TP53` vs `P53`)
- Some false positives may remain
- Rare true genes may be filtered out

This pipeline is a confidence system, not a truth oracle.

---

## 10) Start Checklist

Before running end-to-end:
- Input PDF is readable or OCR-ready
- Section and sentence segmentation works on sample paper
- Candidate extraction and filters are configurable
- Ambiguity list is available
- Scoring threshold is set
- Optional web validation cache is enabled

When all checks pass, run Stage 1 through Stage 7 in order.

# Graph Report - .  (2026-05-09)

## Corpus Check
- Corpus is ~5,482 words - fits in a single context window. You may not need a graph.

## Summary
- 10 nodes · 9 edges · 4 communities (2 shown, 2 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 4,250 input · 1,200 output

## Community Hubs (Navigation)
- [[_COMMUNITY_AMR & POC Diagnostics|AMR & POC Diagnostics]]
- [[_COMMUNITY_Machine Learning Models|Machine Learning Models]]
- [[_COMMUNITY_Raman Spectroscopy Foundations|Raman Spectroscopy Foundations]]
- [[_COMMUNITY_Signal Processing (CWT)|Signal Processing (CWT)]]

## God Nodes (most connected - your core abstractions)
1. `Multimodal Convolutional Neural Network` - 5 edges
2. `Antimicrobial Resistance (AMR)` - 2 edges
3. `Raman Spectroscopy` - 2 edges
4. `Continuous Wavelet Transform (CWT)` - 2 edges
5. `Point-of-Care Diagnostics` - 2 edges
6. `Support Vector Machine (SVM)` - 1 edges
7. `Simple Concatenation` - 1 edges
8. `Morlet Mother Wavelet` - 1 edges
9. `Murray et al. (2022)` - 1 edges
10. `Ho et al. (2019)` - 1 edges

## Surprising Connections (you probably didn't know these)
- `Multimodal Convolutional Neural Network` --processes--> `Raman Spectroscopy`  [EXTRACTED]
  D:/AMR/Confernce_Final_Published (1).md → D:/AMR/Confernce_Final_Published (1).md  _Bridges community 2 → community 1_
- `Multimodal Convolutional Neural Network` --incorporates--> `Continuous Wavelet Transform (CWT)`  [EXTRACTED]
  D:/AMR/Confernce_Final_Published (1).md → D:/AMR/Confernce_Final_Published (1).md  _Bridges community 1 → community 3_
- `Multimodal Convolutional Neural Network` --targets--> `Point-of-Care Diagnostics`  [EXTRACTED]
  D:/AMR/Confernce_Final_Published (1).md → D:/AMR/Confernce_Final_Published (1).md  _Bridges community 1 → community 0_

## Communities (4 total, 2 thin omitted)

### Community 0 - "AMR & POC Diagnostics"
Cohesion: 0.67
Nodes (3): Antimicrobial Resistance (AMR), Murray et al. (2022), Point-of-Care Diagnostics

### Community 1 - "Machine Learning Models"
Cohesion: 0.67
Nodes (3): Multimodal Convolutional Neural Network, Simple Concatenation, Support Vector Machine (SVM)

## Knowledge Gaps
- **5 isolated node(s):** `Support Vector Machine (SVM)`, `Simple Concatenation`, `Morlet Mother Wavelet`, `Murray et al. (2022)`, `Ho et al. (2019)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Multimodal Convolutional Neural Network` connect `Machine Learning Models` to `AMR & POC Diagnostics`, `Raman Spectroscopy Foundations`, `Signal Processing (CWT)`?**
  _High betweenness centrality (0.861) - this node is a cross-community bridge._
- **Why does `Point-of-Care Diagnostics` connect `AMR & POC Diagnostics` to `Machine Learning Models`?**
  _High betweenness centrality (0.389) - this node is a cross-community bridge._
- **What connects `Support Vector Machine (SVM)`, `Simple Concatenation`, `Morlet Mother Wavelet` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
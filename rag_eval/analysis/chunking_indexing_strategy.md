# Chunking & Indexing Improvement Strategy

## Background

Analysis of 5 baseline evaluation runs (KB Batch 1, hybrid search, top_k=5) identified
systematic retrieval failures with near-zero variance across runs (std dev ≈ 0.00 for both
context recall and precision). This confirms the failures are structural — rooted in how
documents are chunked and indexed — rather than random variation.

Three root causes were identified, all addressable at the chunking/indexing stage without
changes to the retrieval architecture.

---

## Issue 1: Stub Chunks — Template & Placeholder Content

### Problem
198 chunks (3.7% of the 5,386-chunk index) contain no useful information. They are
unfilled Confluence CMS templates, navigation headers, or placeholder text that was never
removed when pages were authored. These chunks score high enough in retrieval to occupy
slots in the top-k results while contributing nothing to recall or precision.

### Evidence
| Pattern | Count |
|---------|-------|
| FAQ boilerplate (`"Use an expand section for each FAQ..."`) | 95 |
| Navigation-only (`"Pages on this Topic"`, `"Table of Contents"`, etc.) | 85 |
| Placeholder text (`"Enter answer here"`, `"Enter text here"`, `"(Enter text here)"`) | 59 |
| Previous/Next Steps template (`"links to any procedures that should come before or after..."`) | 13 |
| Section Title placeholder (`"enter the first section's content here"`) | 2 |
| **Total unique stub chunks** | **198** |

### Example
**Tableau FAQs chunk** (retrieved for "What do I need to do to get access to Tableau as a UWF employee?"):
```
# FAQs
Use an expand section for each FAQ that you want to include. Copy and paste the expand
section to create a new question. Make sure to edit the title of the expand section.
Please remove this section if there are no FAQs.
You will have access to Tableau the day after all steps are completed.
```
This chunk occupied one of the 5 retrieval slots with boilerplate template instructions
rather than actual content.

### To-Do
- [ ] Implement a stub chunk filter function that checks `original_content` against the
  identified patterns before indexing. Chunks matching any pattern should be excluded
  from the index entirely.
- [ ] Run the filter against the full chunk file and log how many chunks are removed per
  pattern category.
- [ ] Re-index with stub chunks excluded and re-run the evaluation to measure improvement.

---

## Issue 2: Step-Level Fragmentation

### Problem
Multi-step procedures are chunked at the individual step level (one chunk per step header).
When a user asks a procedural question, the retriever returns a scattered selection of
non-consecutive steps from the same procedure — not a coherent, complete answer.

### Evidence
- **Parking permit purchase** (precision 0.20): top-5 returns Steps 1, 3, 4, and 10 from
  the same document — out of order and non-contiguous.
- **GradesFirst kiosk setup** (recall 0.40): top-5 returns Steps 3, 4, 5, and 10 — skips
  the first two steps entirely.
- **Dual Enrollment Orientation steps**: Retrieved as numbered fragments labeled
  `"incomplete"` (lines 223, 224, 225... 230), clearly a chunking artifact.

### Root Cause
`TextChunker` in `src/knowledge_base/processing/text_chunker.py` currently splits on all
four header levels (`#`, `##`, `###`, `####`) via `MarkdownHeaderTextSplitter`. Every
`### Step N` header creates a new chunk boundary, producing one small chunk per step. After
header splitting, `RecursiveCharacterTextSplitter` (chunk_size=2000, overlap=400) further
splits any section that exceeds the character limit. This second splitter is never reached
for individual steps since they are far below 2000 characters.

### Chosen Approach: Split at H2 (`##`) and Above Only

Remove `###` and `####` from `headers_to_split_on` in `TextChunker.__init__`. All H3/H4
content is then grouped into its parent H2 section. If that combined section exceeds 2000
characters, `RecursiveCharacterTextSplitter` takes over with 400-character overlap.

**What this combines (beneficial):**
- `### Step 1`, `### Step 2`... under `## Instructions` → one chunk containing the full
  procedure. Directly fixes the parking permit and GradesFirst fragmentation.
- `### FAQ Question 1`, `### Question 2`... under `## FAQs` → one chunk containing the
  full FAQ section rather than isolated individual questions.
- H3-level stub chunks (e.g., 3-word Overview sub-sections) get absorbed into their
  parent H2 content, eliminating many stub chunks as a side effect.
- Pages using H3 to separate related sub-topics (e.g., `### Undergraduate`, `### Graduate`
  under `## Academic Programs`) are combined — appropriate since they share topical context.

**Known trade-off:**
For H2 sections with many detailed steps (15+), the combined content will exceed 2000
characters. The `RecursiveCharacterTextSplitter` will then split by character count, and
the split boundary is not semantic — it may cut mid-step. The 400-character overlap
carries context across boundaries, which is still significantly better than the current
approach of returning Steps 1, 3, and 10 with no surrounding context.

### To-Do
- [ ] Modify `headers_to_split_on` in `TextChunker.__init__` to only include `#` and `##`.
- [ ] Re-chunk the full KB and re-index.
- [ ] Re-run evaluation and compare recall/precision scores on procedural questions
  (parking permit, GradesFirst, Dual Enrollment) against the baseline (recall 0.78,
  precision 0.74).

---

## Issue 3: Same-Document Over-Fragmentation

### Problem
Because step-level fragmentation creates many small chunks from a single document, the
retriever can fill all k=5 slots with fragments from that one document. This leaves no
room for other documents that might be needed to fully answer the question.

### Evidence
- **Psychology APC confidentiality** (precision 0.25): 4 of 5 chunks from the same page
  (`Psychology APC Confidentiality Statement`), all different sub-sections.
- **Pearson MyLab + Canvas** (precision 0.37): all 5 chunks from the same page
  (`Pearson MyLab and Mastering`).
- **Parking permit** (precision 0.20): all 5 chunks from the same page
  (`Purchasing a Parking Permit`).

### Relationship to Issue 2
This issue is largely a downstream consequence of step-level fragmentation. A single
procedure page with 10 steps produces 10+ indexable chunks, making it disproportionately
likely to dominate the top-k results. Fixing Issue 2 (larger chunks per section) will
reduce the number of same-document fragments and naturally limit same-document
over-representation.

### To-Do
- [ ] After implementing the chunking strategy changes from Issue 2, re-evaluate whether
  same-document over-fragmentation persists.
- [ ] If it persists after re-chunking, consider adding a document-level deduplication
  step at index time: if multiple chunks from the same source page are created, merge
  or deduplicate them before indexing.

---

## Implementation Order

1. **Issue 1 (Stub Chunk Filtering)** — highest effort-to-impact ratio, purely additive
   preprocessing with no structural changes required.
2. **Issue 2 (Step-Level Fragmentation)** — requires investigating and modifying the
   chunking logic; re-indexing required.
3. **Issue 3 (Same-Document Over-Fragmentation)** — revisit after Issue 2 is addressed,
   since it is likely resolved as a side effect.

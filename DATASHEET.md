# Datasheet for IndoNutriQA

This datasheet follows the *Datasheets for Datasets* framework (Gebru et al.,
2021) to describe the IndoNutriQA dataset: an Indonesian clinical question
answering (QA) dataset focused on the nutritional management of pediatric
and adult inpatients.

---

## Motivation

### For what purpose was the dataset created?

IndoNutriQA was created to support research on retrieval-augmented question
answering (RAG-QA) over Indonesian clinical nutrition records. Existing
clinical QA datasets are dominated by English-language sources and do not
reflect the documentation style, terminology, or care patterns of Indonesian
clinical practice. IndoNutriQA fills this gap by providing curated
SQuAD-style QA pairs grounded in real, anonymized Indonesian case
documents covering nutritional assessment, intervention, and follow-up.

### Who created the dataset and on behalf of which entity?

The dataset was created by the IndoNutriQA project team in collaboration
with clinical nutrition experts. See the accompanying paper and the project
README for authorship and affiliation details.

### Who funded the creation of the dataset?

See the acknowledgments section of the accompanying paper.

---

## Composition

### What do the instances that comprise the dataset represent?

Each instance is a question-answer pair grounded in one or more passages
(chunks) from an anonymized Indonesian clinical nutrition case report.
Instances follow a SQuAD-style schema: a question, an answer, and the
supporting context chunks, with associated metadata identifying the source
document and chunk.

### How many instances are there in total?

See [data/dataset/dataset.json](data/dataset/) for the
authoritative QA file. Counts of source documents, chunks, and QA pairs are
reported in the accompanying paper.

### Does the dataset contain all possible instances or is it a sample?

The dataset is a curated sample of case reports selected to cover a
diverse range of pediatric and adult clinical nutrition scenarios
(e.g., severe acute malnutrition, prematurity, congenital anomalies,
oncology-related cachexia, dengue, pneumonia). It is not intended as a
population-level sample.

### What data does each instance consist of?

Each QA instance contains:

- `question` — a natural language question in Indonesian
- `answer` — the canonical answer string
- `context` — one or more retrieved chunks supporting the answer
- `metadata` — source file identifier, chunk indices, and categorical
  labels (e.g., topic, question type)

Chunks are stored as JSON files under [data/chunks/](data/chunks/)
and are referenced by ID in the QA file.

### Is there a label or target associated with each instance?

Yes — the canonical answer string serves as the supervised target.
Categorical labels (topic, question type) are provided for stratified
analysis.

### Is any information missing from individual instances?

PII has been deliberately removed. See *Preprocessing* below.

### Are relationships between individual instances made explicit?

Yes — QA pairs reference the supporting context chunks by ID, and chunks
reference their source document.

### Are there recommended data splits?

Train / validation / test splits are described in the accompanying paper.

### Are there any errors, sources of noise, or redundancies in the dataset?

Source documents were authored under routine clinical conditions and may
contain typographical errors, abbreviations, or inconsistent formatting
that are preserved in the chunks. The QA pairs were curated to be
faithful to the source text and have undergone validation against the
chunks, but residual annotation noise is possible.

### Is the dataset self-contained, or does it link to external resources?

The dataset is self-contained. Evaluation artifacts refer to specific
LLMs (e.g., GPT-, Claude-, open-weight families) used at the time of
evaluation; reproducing those exact numbers requires API access to
those models.

### Does the dataset contain confidential or sensitive information?

The source documents are clinical case reports that originally contained
PII. All instances in the released dataset have been anonymized using a
custom NER-based pipeline. Users must not attempt to re-identify
individuals.

---

## Collection Process

### How was the data associated with each instance acquired?

Source case reports were collected from clinical nutrition documentation
with appropriate institutional approvals. QA pairs were curated by
domain experts who reviewed the chunks and wrote questions and canonical
answers grounded in the source.

### Over what timeframe was the data collected?

See the accompanying paper.

### Were any ethical review processes conducted?

Source data collection followed the institutional review processes of
the originating clinical site(s). See the accompanying paper for
details.

---

## Preprocessing / Cleaning / Labeling

### Was any preprocessing/cleaning/labeling of the data done?

Yes. The processing pipeline performs:

1. PDF extraction
2. Removal of pre-introduction boilerplate
3. Medical NER extraction
4. PII anonymization using extracted NER entities
5. Matching of curated QA pairs to chunks
6. Conversion of QA pairs to JSON
7. Similarity-based deduplication and validation
8. Retrieval benchmarking
9. Context-based QA evaluation
10. Syntactic divergence analysis
11. Retrieval-augmented QA experiments
12. LLM evaluation
13. Statistical robustness (bootstrap)
14. Semantic evaluation
15. RAGAS evaluation
16. Categorical analysis

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Raw PDF sources are not redistributed. Anonymized chunks are released
under [data/chunks/](data/chunks/).

### Is the software used to preprocess/clean/label the instances available?

Yes — see [src/](src/) and [config/config.py](config/config.py).

---

## Uses

### Has the dataset been used for any tasks already?

The dataset has been used to benchmark retrieval, RAG-QA, and LLM-as-judge
pipelines on Indonesian clinical nutrition QA. See the accompanying paper.

### Is there a repository that links to any or all papers or systems that use the dataset?

See the project README.

### What (other) tasks could the dataset be used for?

- Indonesian clinical NLP (NER, terminology normalization)
- Retrieval evaluation in low-resource clinical domains
- Cross-lingual transfer studies
- LLM evaluation with domain-specific ground truth

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

The dataset reflects the documentation style and case mix of the
originating clinical site(s) and may not generalize to other Indonesian
clinical contexts or languages. Anonymization may have removed
demographic context that is relevant for some research questions.

### Are there tasks for which the dataset should not be used?

The dataset must not be used to:

- Attempt re-identification of individuals
- Make clinical decisions for real patients
- Train commercial diagnostic or triage systems without separate
  clinical validation and regulatory clearance

---

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

Yes — under the CC BY-NC 4.0 license (see [LICENSE-DATA](LICENSE-DATA)). Code is
released under MIT (see [LICENSE-CODE](LICENSE-CODE)).

### How will the dataset be distributed?

Via the project repository.

### When will the dataset be distributed?

See the accompanying paper for the release timeline.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

CC BY-NC 4.0 for data; MIT for code.

---

## Maintenance

### Who will be supporting/hosting/maintaining the dataset?

The IndoNutriQA project team. See the README for contact information.

### How can the owner/curator/manager of the dataset be contacted?

See the README for contact information.

### Will the dataset be updated?

The dataset may be updated to correct errors or to expand coverage.
Versioned releases will be tagged in the repository.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes — via pull requests and issues on the project repository. Contributions
of additional anonymized case reports require clearance with the project
team before integration.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H.,
  Daumé III, H., & Crawford, K. (2021). *Datasheets for datasets*.
  Communications of the ACM, 64(12), 86–92.

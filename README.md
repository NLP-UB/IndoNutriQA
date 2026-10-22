# IndoNutriQA

An Indonesian-language clinical question answering (QA) dataset for the
nutritional management of pediatric and adult inpatients. QA pairs are
provided in SQuAD-style format, grounded in anonymized chunks extracted
from real clinical nutrition case reports.

See [DATASHEET.md](DATASHEET.md) for a full datasheet (Gebru et al., 2021).

## Contents

- **[data/dataset/dataset.json](data/dataset/dataset.json)** — the QA dataset (questions, answers, and the supporting chunk IDs).
- **[data/chunks/](data/chunks/)** — anonymized context chunks, one JSON file per source case report. PII (names, dates, locations, registration numbers, hospital names) is replaced with `<NAMA>`, `<TGL>`, `<LOKASI>`, `<NO_REG>`, `<RS>`, `<NO_PEGAWAI>`, `<NIM>` tokens. Names of treating physicians, supervisors, report authors and presenters are also replaced.
- Additional folders under [data/](data/) contain evaluation artifacts (statistical robustness, semantic and RAGAS evaluations, categorical analyses).

## Dataset format

Each entry in `dataset.json` is a SQuAD-style QA pair:

```json
{
  "question": "...",
  "answer": "...",
  "context_chunks": ["chunk_id_1", "chunk_id_2"],
  "metadata": { ... }
}
```

Chunks referenced by `context_chunks` are stored in [data/chunks/](data/chunks/) and can be loaded by ID.

## Loading the dataset

```python
import json

with open("data/dataset/dataset.json") as f:
    qa_pairs = json.load(f)
```

## Licenses

- **Data**: [CC BY-NC 4.0](LICENSE-DATA) — non-commercial use with attribution.
- **Code**: [MIT](LICENSE-CODE).

## Privacy

Source case reports contained personally identifiable information (PII)
that has been removed via an NER-based anonymization pipeline. **Users
must not attempt to re-identify individuals.** See [DATASHEET.md](DATASHEET.md)
for details.

## Code and reproduction

The saved predictions and results under `data/` are enough to recompute every table and figure of the paper without calling any LLM API again.
The scripts read the dataset from `data/dataset/dataset.json`.

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To call LLM APIs again, create a `.env` file at the project root with `OPENAI_API_KEY=...` and `ANTHROPIC_API_KEY=...`.

### What is not included

- The original report PDFs, the answer spreadsheets (`QnA.xlsx`) and the intermediate extraction steps. They contain patient, hospital and staff identifiers. Pipeline steps 1 to 4c are included only to document how the data was prepared and cannot be re-run from this repository.
- Fine-tuned model checkpoints. The saved predictions in `data/13-statistical_robustness/predictions/` are enough to recompute all scores.

### Layout

```
src/pipelines/        numbered pipeline steps
src/utils/            PDF processing, anonymisation, validation, Hugging Face formatting
config/               shared settings
data/dataset/         the QA dataset
data/chunks/          anonymised context chunks
data/6-... to 20-...  saved predictions and results, one folder per step
docs/rd_annotation/   rubric and labelling set-up for the expert-rater study
notebooks/            demographic and anthropometric figures
```

Run every script from the project root, for example `python src/pipelines/12-llm_evaluation.py --help`.

### Pipeline steps

| Step | Script | What it does |
|---|---|---|
| 1-4c | `1-run_extraction.py` ... `4c-anonymize_staff_names.py` | PDF extraction, chunking, removal of patient identifiers (4), hospital and publisher names (4b), and staff, author and student names and numbers (4c) |
| 5-7 | `5-match_qna_to_chunks.py`, `6-xlsx_to_qa_json.py`, `7-json_file_similarity.py` | Match each answer to its source chunk and build the dataset file |
| 8 | `8-retrieval_benchmarking.py` | Retrieval benchmark |
| 9 | `9-context_based_QA.py` | Fine-tuned extractive and generative baselines |
| 10 | `10-syntactic_divergence.py` | Edit distance between predictions and references |
| 11 | `11-retrieval_augmented_QA.py` | RAG baseline |
| 12 | `12-llm_evaluation.py` | Zero- and five-shot LLM evaluation |
| 13 | `13-statistical_robustness.py` | Bootstrap confidence intervals, cross-validation, learning curve |
| 14 | `14-semantic_evaluation.py` | BERTScore, embedding cosine, LLM-as-judge, self-preference test |
| 15 | `15-ragas_evaluation.py` | RAGAS diagnostic for the RAG baseline |
| 16 | `16-categorical_analysis.py` | Results by question type, disease category, and error taxonomy |
| 17 | `17-visualization.py` | Figures from the saved results |
| 18 | `18-rd_judge_agreement.py` | Expert-rater study and judge-vs-rater agreement |
| 20 | `20-human_error_analysis.py` | Coding of the raters' free-text notes |

### Paper tables and result files

| Paper part | Script | Result file |
|---|---|---|
| Model performance | `9-`, `11-`, `12-` | `data/9-context_based_QA/`, `data/12-llm_evaluation/llm_results.json` |
| Cross-validation, learning curve, bootstrap CIs | `13-` | `data/13-statistical_robustness/` |
| Semantic metrics, LLM-as-judge, self-preference | `14-` | `data/14-semantic_evaluation/` |
| RAGAS | `15-` | `data/15-ragas_evaluation/ragas_results.json` |
| Question type, disease category, error taxonomy | `16-` | `data/16-categorical_analysis/` |
| Expert-rater agreement | `18-` | `data/18-rd_judge_agreement/agreement.json` |
| Human error analysis | `20-` | `data/20-human_error_analysis/human_errors.json` |

The comparison tables with other benchmarks were compiled by hand from the original papers and have no script.

## Citation

If you use IndoNutriQA, please cite the accompanying paper (citation
details to follow).

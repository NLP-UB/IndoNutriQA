import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class IndoBERTExtractiveQA:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

nltk.download("punkt")

# ----------------------------------------------------
# 1. Syntactic divergence function (token-level)
# ----------------------------------------------------
def syntactic_divergence(prediction, reference):
    """
    Normalized token-level edit distance.
    0   = identical syntax
    >0  = increasing syntactic divergence
    """
    pred_tokens = word_tokenize(prediction.lower())
    ref_tokens = word_tokenize(reference.lower())

    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 0.0
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0

    dist = edit_distance(pred_tokens, ref_tokens)
    return dist / max(len(pred_tokens), len(ref_tokens))

def predict(question, context, max_length=512):
    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=-1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=-1).item() + 1

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx],
        skip_special_tokens=True
    )
  


    return answer.strip()



# ----------------------------------------------------
# 2. Load dataset
# ----------------------------------------------------
with open("data/dataset/dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} QA pairs")


# ----------------------------------------------------
# 3. Compute syntactic divergence for all samples
# ----------------------------------------------------
syntactic_divergences = []

for item in data:
    question = item["question"]
    context = item["context"]

    # gold answer
    reference = item["answer"]["text"][0]



    MODEL_CHECKPOINT = "models/rifky_indobert_qa_finetuned/checkpoint-246"
    TOKENIZER_PATH   = "models/rifky_indobert_qa_finetuned"  # or base model

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Loaded successfully")


    # model prediction (replace with your actual model)
    # prediction = model.predict(question, context)

    prediction = predict(question, context)

    d = syntactic_divergence(prediction, reference)
    syntactic_divergences.append(d)

syntactic_divergences = np.array(syntactic_divergences)


# ----------------------------------------------------
# 4. Plot histogram (percentage)
# ----------------------------------------------------
plt.figure()

plt.hist(
    syntactic_divergences,
    bins=10,
    weights=np.ones(len(syntactic_divergences)) / len(syntactic_divergences) * 100
)

plt.xlabel("Syntactic divergence")
plt.ylabel("Percentage")
# plt.title("(a) Histogram of syntactic divergence")

plt.tight_layout()
plt.savefig("data/10-syntactic_divergence/syntactic_divergence_histogram.pdf", format="pdf", bbox_inches="tight")
plt.show()

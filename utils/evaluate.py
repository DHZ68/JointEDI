import torch
from tqdm import tqdm
import time
import json
import re
from collections import Counter
from collections import defaultdict


def extract(input_string):
    Candidate_content = re.search(r"Candidate: (.*?),", input_string)
    euphemism_label_content = re.search(r"Euphemism Label: (.*?),", input_string)
    class_label_content = re.search(r"Class Label: (.*?)[)]", input_string)

    candidate = Candidate_content.group(1) if Candidate_content else None
    euphemism_label = (
        euphemism_label_content.group(1) if euphemism_label_content else None
    )
    class_label = class_label_content.group(1) if class_label_content else None
    #
    return (euphemism_label, class_label)

def compute_metrics_pair(pred, tgt):
    """
    The form of pre and tgt is as follows: (bi, cls).
    bi indicates that the candidate word in the sentence uses euphemism.
    cls indicates which category the euphemism meaning represented by the candidate word belongs to.
    """
    tp = fp = fn = pair_tp = pair_fp = pair_fn = 0
    for pred_tuple, tgt_tuple in zip(pred, tgt):
        if pred_tuple[0] == "1" and tgt_tuple[0] == "1":
            tp += 1
            if pred_tuple[1] == tgt_tuple[1]:
                pair_tp += 1
            else:
                pair_fn += 1
                pair_fp += 1
        elif pred_tuple[0] == "1" and tgt_tuple[0] == "0":
            fp += 1
            pair_fp += 1
        elif pred_tuple[0] !="1" and tgt_tuple[0] == "1":
            fn += 1
            pair_fn += 1

    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (pre * recall) / (pre + recall) if (pre + recall) > 0 else 0

    # Calculating pair metrics
    pair_precision = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) > 0 else 0
    pair_recall = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) > 0 else 0
    pair_f1 = (
        2 * (pair_precision * pair_recall) / (pair_precision + pair_recall)
        if (pair_precision + pair_recall) > 0
        else 0
    )

    # Calculating identification metrics

    return f1, recall, pre, pair_f1, pair_recall, pair_precision

def evaluate(model, tokenizer, data_loader, device):
    print("\n[Evaluating...]")
    generated_text_file = "generated_text.json"
    reference_text_file = "reference_text.json"
    model.eval()
    all_generated_text = []  # Store generated text
    all_reference_text = []  # Store reference text
    start_time = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss
            total_loss += loss.item()
            generated_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_length=128
            )
            generated_text = [
                tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
            ]
            reference_text = [
                tokenizer.decode(t, skip_special_tokens=True) for t in batch["labels"]
            ]
            all_generated_text.extend(generated_text)
            all_reference_text.extend(reference_text)

    if len(all_generated_text) != len(all_reference_text):
        raise ValueError("List lengths for generated text and reference text are inconsistent.")

    all_generated_tuple = []
    all_reference_tuple = []
    for generated_text, reference_text in zip(all_generated_text, all_reference_text):
        all_generated_tuple.append(extract(generated_text))
        all_reference_tuple.append(extract(reference_text))

    with open(generated_text_file, "w", encoding="utf-8") as json_file:
        json.dump(all_generated_tuple, json_file, ensure_ascii=False, indent=4)

    with open(reference_text_file, "w", encoding="utf-8") as json_file:
        json.dump(all_reference_tuple, json_file, ensure_ascii=False, indent=4)
    f1, rec, pre, pair_f1, pair_rec, pair_pre = compute_metrics_pair(
        all_generated_tuple, all_reference_tuple
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation Time: {elapsed_time:.2f} seconds")
    average_loss = total_loss / len(data_loader)
    return average_loss, f1, rec, pre, pair_f1, pair_rec, pair_pre

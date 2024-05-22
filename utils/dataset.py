import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


class EuphemismDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer
        self.max_seq_len = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get a single data item
        item = self.data[idx]
        sentence = item["mask_sentence"]
        candidate = item["candidate"]
        euphemism_label = item["bi_label"]
        class_label = item["label"]
        input_text = (
            f"Task: Euphemism Detection and Identification\nSentence: {sentence}"
        )
        target_text = (
            f"Target: (Euphemism Label: {euphemism_label}, Class Label: {class_label})"
        )
        target_text1 = f"Target: Euphemism Label: {euphemism_label}"
        target_text2 = f"Target: Class Label: {class_label}"
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encoding1 = self.tokenizer(
            target_text1,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encoding2 = self.tokenizer(
            target_text2,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "labels1": target_encoding1["input_ids"].squeeze(),
            "labels2": target_encoding2["input_ids"].squeeze(),
        }


def custom_collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    labels1 = [item["labels1"] for item in batch]
    labels2 = [item["labels2"] for item in batch]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    if isinstance(labels[0], torch.Tensor):
        if len(labels[0].shape) > 0:
            labels = pad_sequence(labels, batch_first=True, padding_value=0)
        else:
            labels = torch.stack(labels)

    if isinstance(labels1[0], torch.Tensor):
        if len(labels1[0].shape) > 0:
            labels1 = pad_sequence(labels1, batch_first=True, padding_value=0)
        else:
            labels1 = torch.stack(labels1)

    if isinstance(labels2[0], torch.Tensor):
        if len(labels2[0].shape) > 0:
            labels2 = pad_sequence(labels2, batch_first=True, padding_value=0)
        else:
            labels2 = torch.stack(labels2)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels1": labels1,
        "labels2": labels2,
    }


class CustomMBartForConditionalGeneration(MBartForConditionalGeneration):
    """_summary_

    Args:
        MBartForConditionalGeneration (_type_): _description_
        implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, labels=None, task_id=0, **kwargs):
        outputs = super().forward(
            input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )
        loss = outputs.loss if labels is not None else None
        return Seq2SeqLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

import torch
import torch.nn as nn
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer,
    MT5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.dataset import (
    EuphemismDataset,
    custom_collate_fn,
    CustomMBartForConditionalGeneration,
)
from torch.utils.data import DataLoader
from utils.work import model_train, model_test
from utils.log import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="Chinese",
    choices=[
        "English",
        "Chinese",
    ],
)  # dataset file name
parser.add_argument(
    "--model_name",
    type=str,
    default="facebook/mbart-large-cc25",
    choices=[
        "facebook/mbart-large-cc25",
        "facebook/mbart-large-50",
        "google/mt5-base",
        "google/mt5-large",
    ],
)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument(
    "--random_seed", type=int, default=3407, choices=[3407, 1334, 7, 5, 42]
)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--gamma", type=float, default=0.2)
args = parser.parse_args()
args_dict = vars(args)
save_config(args_dict, log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
random_seed = args.random_seed
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
alpha = args.alpha
beta = args.beta
gamma = args.gamma
model_name = args.model_name
dataset = args.dataset
train_path = f"data/{dataset}/train.json"
dev_path = f"data/{dataset}/dev.json"
test_path = f"data/{dataset}/test.json"
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
save_model = args.save_model
logger.info(
    f"[Model Name: {model_name} - DataSet: {dataset} - Random Seed: {random_seed}]"
)
lang_dict = {
    "English": "en_XX",
    "Chinese": "zh_CN",
}

print("[Model initialing...]")
if model_name.startswith("facebook"):
    model = CustomMBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = lang_dict[dataset]
    tokenizer.tgt_lang = lang_dict[dataset]
elif model_name.startswith("google"):
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
model.to(device)
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

dataset_train = EuphemismDataset(tokenizer, train_path)
dataset_dev = EuphemismDataset(tokenizer, dev_path)
dataset_test = EuphemismDataset(tokenizer, test_path)
print(len(dataset_train), len(dataset_dev), len(dataset_test))

print("[Creating Dataloader...]")
train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
)
dev_loader = DataLoader(
    dataset_dev,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
)
test_loader = DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: custom_collate_fn(batch, tokenizer),
)

logger.info(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")
if args.load_model is None:
    model_path = model_train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        save_model=save_model,
        dataset=dataset,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        device=device,
    )
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        model.to(device)

else:
    model_path = args.load_model
    model.load_state_dict(torch.load(model_path))
    model.to(device)

model_test(model, tokenizer, test_loader, device)

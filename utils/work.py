from tqdm import tqdm
from utils.evaluate import evaluate
import numpy as np
import torch.nn.utils as nn_utils
import torch.nn as nn
import time
import os
import torch
from utils.log import time_logger, logger
from sklearn.model_selection import KFold
from utils.dataset import custom_collate_fn
from torch.optim.lr_scheduler import CosineAnnealingLR

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def format_time(seconds):
    """Format the time into the form "h:m:s""""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"


def model_test(model, tokenizer, test_loader, device):
    print("[Start testing...]")
    _, f1, rec, pre, pair_f1, pair_rec, pair_pre = evaluate(
        model, tokenizer=tokenizer, data_loader=test_loader, device=device
    )
    logger.info(
        f"Test F1 Score: {f1:.4f} "
        f"- Recall: {rec:.4f} "
        f"- Precision: {pre:.4f} "
        f"- Pair F1 Score: {pair_f1:.4f} "
        f"- Pair Recall: {pair_rec:.4f} "
        f"- Pair Precision: {pair_pre:.4f}"
    )


def model_train(
    model,
    tokenizer,
    train_loader,
    dev_loader,
    num_epochs,
    optimizer,
    save_model,
    dataset,
    alpha,
    beta,
    gamma,
    device,
):
    print("[Start training...]")
    start_time = time.time()
    new_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    time_logger.info(f"Start Training: {new_start_time}")
    model_time = time.strftime("%Y%m%d-%H%M%S")
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_score = torch.inf
    no_improvement_count = 0
    patience = 4
    total_steps = len(train_loader) * num_epochs
    global_step = 0
    eps = 1e-8
    model_path = None
    with tqdm(total=total_steps, desc="Training", dynamic_ncols=True) as pbar:
        for epoch in range(num_epochs):
            model.train()
            avg_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                labels1 = batch["labels1"].to(device)
                labels2 = batch["labels2"].to(device)
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                outputs1 = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels1
                )
                outputs2 = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels2
                )
                loss_main = outputs.loss
                loss_aux1 = outputs1.loss
                loss_aux2 = outputs2.loss
                loss_total = alpha * loss_main + beta * loss_aux1 + gamma * loss_aux2
                loss_total.backward()
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_loss += loss_total.item()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_total.item():.5f}")
            scheduler.step()
            avg_loss /= len(train_loader)
            val_loss, f1, rec, pre, pair_f1, pair_rec, pair_pre = evaluate(
                model, tokenizer=tokenizer, data_loader=dev_loader, device=device
            )
            logger.info(
                f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f} "
                f"Validation loss: {val_loss:.4f} "
                f"- F1 Score: {f1:.4f} "
                f"- Recall: {rec:.4f} "
                f"- Precision: {pre:.4f} "
                f"- Pair F1 Score: {pair_f1:.4f} "
                f"- Pair Recall: {pair_rec:.4f} "
                f"- Pair Precision: {pair_pre:.4f}"
            )
            # early stopping
            if val_loss < best_score:
                best_score = val_loss
                no_improvement_count = 0
                save_model_dir = f"save_model/{dataset}"
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                model_path = f"{save_model_dir}/model_{model_time}.pt"
                torch.save(model.state_dict(), model_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info(
                        "Early Stopping: No improvement in validation Pair F1 score for {} epochs. Training stopped.".format(
                            patience
                        )
                    )
                    break
    end_time = time.time()
    new_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    time_logger.info(f"End Training: {new_end_time}")
    elapsed_time = end_time - start_time
    time_logger.info(f"Training Time: {format_time(elapsed_time)}")
    # save model
    if save_model:
        save_model_dir = f"save_model/{dataset}"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        model_path = f"{save_model_dir}/model_{model_time}.pt"
        torch.save(model.state_dict(), model_path)
    return model_path

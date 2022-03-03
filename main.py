#!/usr/bin/env python
# coding: utf-8
from asyncio.log import logger
from curses.ascii import EM
import os
from re import I
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import numpy as np
from pathlib import Path
import random
import shlex
import subprocess
import pytorch_lightning as pl

import datasets
import torch
import torch.utils.data 
import torchmetrics
import transformers
import wandb
import wget

try:
    import colored_traceback.auto
except ImportError:
    pass

def cmd(cmd):
    if isinstance(cmd, list):
        cmd = shlex.join(cmd)

    return subprocess.check_output(cmd, shell=True).strip().decode().split("\n")


def maybe_download(path, url, md5):
    if Path(path).exists():
        computed_md5 = cmd(["md5sum", path])[0].split()[0]
        if not computed_md5 == md5:
            print(
                f"Expected: '{md5}', {len(md5)}\n"
                f"Got:      '{computed_md5}', {len(md5)}"
            )
            raise ValueError(
                f"md5 mismatch for {path}\n."
            )

    if not Path(path).exists():
        wget.download(url, path)


def load_dataset(path, tokenizer):
    all_lines = Path(path).read_text().strip().split("\n")
    inputs = []
    labels = []
    for input_val, labels_val, _ in [x.split("\t") for x in all_lines]:
        inputs.append(input_val)
        labels.append(labels_val)

    return inputs, labels


def prepare_ds(tokenizer, x, y):
    ds = datasets.Dataset.from_dict(dict(x=x, y=y))
    ds = ds.map(lambda example: tokenizer(example["x"], truncation=True), remove_columns=["x"])
    ds = ds.map(lambda example: dict(labels=tokenizer(example["y"], truncation=True)["input_ids"]), remove_columns=["y"])
    return ds


class OurMetric:
    @classmethod
    def prepare(cls, x, tokenizer, pred, label, do_print):
        things_to_ignore = {
            -100, 
            tokenizer.pad_token_id, 
            tokenizer.bos_token_id, 
            tokenizer.eos_token_id
        }
        
        assert 0 in things_to_ignore, things_to_ignore
        assert 1 in things_to_ignore, things_to_ignore
        assert 2 in things_to_ignore, things_to_ignore


        cleaned_preds = [x for x in  pred.cpu().numpy().tolist() if x not in things_to_ignore],
        cleaned_labels = [x for x in label.cpu().numpy().tolist() if x not in things_to_ignore]

        if do_print:
            print("#" * 80)
            print("Preds:  ", cleaned_preds)
            print("Labels: ", cleaned_labels)

        return dict(
            cleaned_preds=cleaned_preds, 
            cleaned_labels=cleaned_labels
        )

    def add(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")

    def compute(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")


class EM(OurMetric):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, pred, label, do_print=False):
        prepped_decoded = list(pred)
        prepped_label =   list(label)

        if do_print:
            print(f"{prepped_decoded = }")
            print(f"{prepped_label =   }")

        if prepped_decoded == prepped_label:
            self.exact_matches += 1
        self.number_seen += 1 
    
    def compute(self, *args, **kwargs):
        return self.correct / self.total


class RecallAcc:
    def __init__(self):
        self.recall_accuracies = []

    def add(self, pred, label, do_print=False):
        recall_acc_decoded = list(pred)
        recall_acc_label = list(label)

        if len(recall_acc_decoded) < len(recall_acc_label):
            recall_acc_decoded += [0] * (len(recall_acc_label) - len(recall_acc_decoded))
        elif len(recall_acc_decoded) > len(recall_acc_label):
            recall_acc_decoded = recall_acc_decoded[:len(recall_acc_label)]

        recall_acc_label_np =    np.array(recall_acc_label,   dtype=np.int64)
        recall_acc_decoded_np =  np.array(recall_acc_decoded, dtype=np.int64)
        recall_acc =             np.mean(recall_acc_decoded_np == recall_acc_label_np)

        self.recall_accuracies.append(recall_acc)

    def compute(self):
        return np.mean(self.recall_accuracies)


class PrecisionAcc:
    def __init__(self): 
        self.precision_accuracies = []

    def add(self, pred, label):
        precision_acc_decoded = list(pred)
        precision_acc_label = list(label)

        if len(precision_acc_decoded) > len(precision_acc_label):
            precision_acc_label += [0] * (len(precision_acc_decoded) - len(precision_acc_label))
        elif len(precision_acc_decoded) < len(precision_acc_label):
            precision_acc_label = precision_acc_label[:len(precision_acc_decoded)]

        precision_acc_label_np =    np.array(precision_acc_label,   dtype=np.int64)
        precision_acc_decoded_np =  np.array(precision_acc_decoded, dtype=np.int64)
        precision_acc =             np.mean(precision_acc_decoded_np == precision_acc_label_np) 

        self.precision_accuracies.append(precision_acc) 

    def compute(self):
        return np.mean(self.precision_accuracies)


class PLBart(pl.LightningModule):
    def __init__(self, *, 
            model, 
            tokenizer, 
            train_ds, 
            eval_ds, 
            gen_ds, 
            train_batch_size, 
            eval_batch_size, 
            num_workers_dl, 
            generation_kwargs,
            seed,
        ):
        super().__init__()
        self.model = model
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.gen_ds = gen_ds
        self.tokenizer = tokenizer
        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers_dl = num_workers_dl
        self.generation_kwargs = generation_kwargs
        self.learning_rate = 1e-4
        self.logging_conf = dict(prog_bar=True, on_step=True, on_epoch=True, logger=True)

        self.shuffle_seed_train = seed
        self.shuffle_seed_eval = seed + 1
        self.shuffle_seed_gen = seed + 2

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train_loss", loss, **self.logging_conf)
        return loss

    def validation_step(self, batch_package, batch_idx):
        for name, batch in batch_package.items():
            loss = self(**batch).loss

            preds = self.model.generate(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                **self.generation_kwargs, 
            )

            em = EM()
            recall_accuracy = RecallAcc()
            precision_accuracy = PrecisionAcc()
            
            for i, (x, pred, label) in enumerate(zip(batch["input_ids"], preds, batch["labels"])):
                do_print = batch_idx == 0 and i == 0
                cleaned = OurMetric.prepare(
                    x, self.tokenizer, pred, label, do_print=do_print
                )

                clean_pred = cleaned["cleaned_preds"]
                clean_label = cleaned["cleaned_labels"]

                em                .add(clean_pred, clean_label, do_print=do_print)
                recall_accuracy   .add(clean_pred, clean_label, do_print=do_print)
                precision_accuracy.add(clean_pred, clean_label, do_print=do_print)

            em_acc_val =         em.compute()
            recall_acc_val =     recall_accuracy.compute()
            precision_acc_val =  precision_accuracy.compute()
            f1_ACC =             2 * precision_acc_val * recall_acc_val / (precision_acc_val + recall_acc_val)
            
            self.log(f"{name}_loss",           loss,               **self.logging_conf)
            self.log(f"{name}_EM",             EM,                 **self.logging_conf)
            self.log(f"{name}_recall_ACC",     recall_acc_val,     **self.logging_conf)
            self.log(f"{name}_precision_ACC",  precision_acc_val,  **self.logging_conf)
            self.log(f"{name}_f1_ACC",         f1_ACC,             **self.logging_conf)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def make_dataloader(self, ds, batch_size):
        return torch.utils.data.DataLoader(
            ds, 
            collate_fn=transformers.data.data_collator.DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True
            ), 
            batch_size=batch_size, 
            num_workers=self.num_workers_dl,
        ) 

    def train_dataloader(self):
        return self.make_dataloader(self.train_ds, self.batch_size).shuffle(self.shuffle_seed_train)

    def val_dataloader(self):
        return dict(
            eval=self.make_dataloader(
                self.eval_ds.shuffle(seed=self.shuffle_seed_eval),
                self.eval_batch_size
            ), 
            gen=self.make_dataloader(
                self.eval_ds.shuffle(seed=self.shuffle_seed_gen),
                self.eval_batch_size
            ), 
        )

TRAIN_PATH = "./train.tsv"
TRAIN_URL = "https://github.com/najoungkim/COGS/blob/main/data/train.tsv?raw=true"
TRAIN_MD5 = "063d79fdfcacf8b04c64d430f7da6717"

EVAL_PATH = "./dev.tsv"
EVAL_URL = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/dev.tsv"
EVAL_MD5 = "69ab5bf9425339f24a732785a0982744"

GEN_PATH = "./gen.tsv"
GEN_URL = "https://github.com/najoungkim/COGS/blob/main/data/gen.tsv?raw=true"
GEN_MD5 = "e6d4a859a25af9ba3319b2a27815a181"

NUM_WORKERS_DL = int(cmd("nproc")[0])
MODEL_NAME = "facebook/bart-base"

TRAIN_MAX_EPOCHS = 80
NUM_TOTAL_BATCH_SEEN = 64 * 8
TRAIN_BATCH_SIZE = 64 * 2
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE
ACCUMULATE_GRAD_BATCHES = round(NUM_TOTAL_BATCH_SEEN / TRAIN_BATCH_SIZE)
assert NUM_TOTAL_BATCH_SEEN % TRAIN_BATCH_SIZE == 0, (
    f"{NUM_TOTAL_BATCH_SEEN % TRAIN_BATCH_SIZE} != 0"
)


GENERATION_KWARGS = dict(
    num_beams=4
)

PROJECT_NAME = "cogs_curriculum"
RUN_NAME = "FIRST"
VAL_CHECK_INTERVAL = 60
LOG_EVERY_N_STEPS = 1
LIMIT_VAL_BATCHES = 4

WANDB_ENTITY = "julesgm"
RANDOM_SEED = 42


def main():

    # These are tiny DS, probably
    maybe_download(TRAIN_PATH, TRAIN_URL, TRAIN_MD5)
    maybe_download(EVAL_PATH, EVAL_URL, EVAL_MD5)
    maybe_download(GEN_PATH, GEN_URL, GEN_MD5)

    model_name = MODEL_NAME
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    train_x, train_y = load_dataset(TRAIN_PATH, tokenizer)
    eval_x, eval_y = load_dataset(EVAL_PATH, tokenizer)
    gen_x, gen_y = load_dataset(GEN_PATH, tokenizer)

    train_ds = prepare_ds(tokenizer, train_x, train_y)
    eval_ds = prepare_ds(tokenizer, eval_x, eval_y)
    gen_ds = prepare_ds(tokenizer, gen_x, gen_y)

    pl_model = PLBart(
        model=model, 
        tokenizer=tokenizer, 
        train_ds=train_ds, 
        eval_ds=eval_ds,
        gen_ds=gen_ds,
        train_batch_size=TRAIN_BATCH_SIZE, 
        eval_batch_size=EVAL_BATCH_SIZE, 
        num_workers_dl=NUM_WORKERS_DL, 
        generation_kwargs=GENERATION_KWARGS,
        seed=RANDOM_SEED,
    )
    trainer = pl.Trainer(
        max_epochs=TRAIN_MAX_EPOCHS, 
        accelerator="gpu", 
        devices=1, 
        logger=pl.loggers.WandbLogger(
            project=PROJECT_NAME, 
            name=RUN_NAME, 
            log_model=False, 
            entity=WANDB_ENTITY
        ),
        val_check_interval=VAL_CHECK_INTERVAL,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        limit_val_batches=LIMIT_VAL_BATCHES,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    )
    trainer.fit(pl_model)


if __name__ == "__main__":
    main()


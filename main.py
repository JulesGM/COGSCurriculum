#!/usr/bin/env python
# coding: utf-8
import abc
from dataclasses import dataclass, field
import itertools 
import json
import os
from pathlib import Path
import shlex
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import allennlp
from beartype import beartype
import datasets
import fire
import numpy as np
import pytorch_lightning as pl
import rich
import torch
import torch.utils.data 
import transformers
import wandb
import wget

try:
    import colored_traceback.auto
except ImportError:
    pass

import allen_ai_bart

###############################################################################
# Table of contents:
###############################################################################
# 1. Varia utils.
# 2. Dataset utils.
# 3. Metrics.
# 4. Lightning module.
# 5. Config.
# 6. Main function.
###############################################################################


###############################################################################
# Varia utils.
###############################################################################
def cmd(command):
    """Execute a command in a subprocess and clean up the output.
    """
    if isinstance(command, list):
        command = shlex.join(command)
    return subprocess.check_output(
        command, shell=True
    ).strip().decode().split("\n")

def get_nprocs():
    num_cmd = int(cmd(["nproc"])[0])
    num_os = len(os.sched_getaffinity(0))
    assert num_cmd == num_os, (num_cmd, num_os)
    return num_cmd

def md5_file(path):
    return cmd(["md5sum", path])[0].split()[0]


###############################################################################
# Dataset utils.
###############################################################################
def maybe_download(path, url, md5_):
    """ Download a file if it's not locally present, raise if the md5 doesn't match.
    """
    if Path(path).exists():
        computed_md5 = md5_file(path)
        if not computed_md5 == md5_:
            print(
                f"Expected: '{md5_}', {len(md5_)}\n"
                f"Got:      '{computed_md5}', {len(md5_)}"
            )
            raise ValueError(
                f"md5 mismatch for {path}\n."
            )

    if not Path(path).exists():
        wget.download(url, path)


def load_dataset(path):
    all_lines = Path(path).read_text().strip().split("\n")
    inputs = []
    labels = []
    for input_val, labels_val, _ in [x.split("\t") for x in all_lines]:
        inputs.append(input_val)
        labels.append(labels_val)

    return inputs, labels


def prepare_ds(tokenizer, x, y):
    ds = datasets.Dataset.from_dict(dict(x=x, y=y))
    ds = ds.map(
        lambda example: tokenizer(example["x"], truncation=True), 
        remove_columns=["x"],
    )
    ds = ds.map(
        lambda example: dict(labels=tokenizer(example["y"], truncation=True)["input_ids"]), 
        remove_columns=["y"],
    )
    return ds


###############################################################################
# Metrics.
###############################################################################
class OurMetric(abc.ABC):
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

        cleaned_preds = [x for x in  pred.cpu().numpy().tolist() if x not in things_to_ignore]
        cleaned_labels = [x for x in label.cpu().numpy().tolist() if x not in things_to_ignore]

        # if do_print:
        #     print("#" * 80)
        #     print("Preds:  ", cleaned_preds)
        #     print("Labels: ", cleaned_labels)

        return dict(
            cleaned_preds=cleaned_preds, 
            cleaned_labels=cleaned_labels
        )

    @abc.abstractmethod
    def add(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")

    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        raise RuntimeError("Shouldn't be run directly")


class EM(OurMetric):
    def __init__(self):
        self.total = 0
        self.correct = 0

    def add(self, pred: list, label: list, do_print=False):
        prepped_decoded = list(pred)
        prepped_label =   list(label)

        is_match = prepped_decoded == prepped_label
        is_match_np = np.all(np.array(prepped_decoded) == np.array(prepped_label)) 
        assert is_match == is_match_np, (is_match, is_match_np)

        if prepped_decoded == prepped_label:
            # rich.print("[green]Exact Match!")
            self.correct += 1
        else:
            # rich.print("[red]Not an Exact Match!")
            pass
        
        if do_print:            
            rich.print(f"prepped_decoded: " + ", ".join(
                [f"[green]{a}" if a == b else f"[red]{a}" for a, b 
                in itertools.zip_longest(prepped_decoded, prepped_label, fillvalue="<None>")]
            ))
            rich.print(f"prepped_label:   " + ", ".join(
                [f"[green]{b}" if a == b else f"[red]{b}" for a, b 
                in itertools.zip_longest(prepped_decoded, prepped_label, fillvalue="<None>")]
            ))

        self.total += 1 
    
    def compute(self, *args, **kwargs):
        rich.print(f"[orange]GOT {self.correct}/{self.total} = {self.correct/self.total:.2f}")
        return self.correct / self.total


class RecallAcc:
    def __init__(self):
        self.recall_accuracies = []

    @beartype
    def add(self, pred: list, label: list, do_print: bool = False):
        recall_acc_decoded = list(pred)
        recall_acc_label = list(label)

        if len(recall_acc_decoded) < len(recall_acc_label):
            recall_acc_decoded += [0] * (len(recall_acc_label) - len(recall_acc_decoded))
        elif len(recall_acc_decoded) > len(recall_acc_label):
            recall_acc_decoded = recall_acc_decoded[:len(recall_acc_label)]

        recall_acc_label_np =   np.array(recall_acc_label,   dtype=np.int64)
        recall_acc_decoded_np = np.array(recall_acc_decoded, dtype=np.int64)
        recall_acc =            np.mean(recall_acc_decoded_np == recall_acc_label_np)

        self.recall_accuracies.append(recall_acc)

    def compute(self):
        return np.mean(self.recall_accuracies)


class PrecisionAcc:
    def __init__(self): 
        self.precision_accuracies = []

    @beartype
    def add(self, pred: list, label: list, do_print: bool = False):
        precision_acc_decoded = list(pred)
        precision_acc_label = list(label)

        if len(precision_acc_decoded) > len(precision_acc_label):
            precision_acc_label += [0] * (len(precision_acc_decoded) - len(precision_acc_label))
        elif len(precision_acc_decoded) < len(precision_acc_label):
            precision_acc_label = precision_acc_label[:len(precision_acc_decoded)]

        precision_acc_label_np =   np.array(precision_acc_label,   dtype=np.int64)
        precision_acc_decoded_np = np.array(precision_acc_decoded, dtype=np.int64)
        precision_acc =            np.mean(precision_acc_decoded_np == precision_acc_label_np) 

        self.precision_accuracies.append(precision_acc) 

    def compute(self):
        return np.mean(self.precision_accuracies)


###############################################################################
# Lightning Module.
###############################################################################
class PLBart(pl.LightningModule):
    def __init__(self, *, 
            use_label_smoothing,
            model, 
            tokenizer, 
            train_ds, 
            eval_ds, 
            gen_ds, 
            train_batch_size, 
            eval_batch_size, 
            num_workers_dl, 
            generation_kwargs,
            learning_rate,
            is_adamw,
            weight_decay,
            scheduler_type,
            scheduler_kwargs,
        ):
        super().__init__()
        self.tokenizer =            tokenizer
        self.model =                model 
        self.use_label_smoothing =  use_label_smoothing
        self.batch_size =           train_batch_size
        self.eval_batch_size =      eval_batch_size
        self.generation_kwargs =    generation_kwargs
        self.logging_conf =         dict(
            prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

        # Related to datasets
        self.train_ds =             train_ds
        self.eval_ds =              eval_ds
        self.gen_ds =               gen_ds
        self.num_workers_dl =       num_workers_dl

        # Things required for auto curriculum:
        self.compute_pg =           False
        self.active_batch =         None
        self.active_true_loss =     None

        # Specific to the optimizer:
        self.learning_rate =        learning_rate
        self.is_adamw =             is_adamw
        self.weight_decay =         weight_decay

        # Related to the scheduler:
        self.scheduler_type =       scheduler_type
        self.scheduler_kwargs =     scheduler_kwargs

        # Experimental
        self.allen_ai_bart =        allen_ai_bart.BartReuser(
            model=self.model, 
            model_name=self.tokenizer.name_or_path, 
            vocab=allennlp.data.Vocabulary.from_pretrained_transformer(self.tokenizer.name_or_path), 
            max_decoding_steps=generation_kwargs["max_length"],
            beam_size=generation_kwargs["num_beams"],
        )

        if self.use_label_smoothing:
            self.label_smoother = transformers.trainer_pt_utils.LabelSmoother()
            assert self.label_smoother.epsilon == 0.1, self.label_smoother.epsilon
        else:
            self.label_smoother = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, **self.logging_conf)

        if self.compute_pg:
            self.active_batch = batch
            self.active_true_loss = outputs.loss

        if self.use_label_smoothing:
            loss = self.label_smoother(outputs, batch["labels"])
        else:
            loss = outputs.loss

        return loss


    def backward(self, loss, optimizer, optimizer_idx):
        super().backward(loss, optimizer, optimizer_idx)

        if self.compute_pg:
            # Active true loss is the loss for the current batch
            # without label smoothing.
            pg = self.active_true_loss - self.model(**self.active_batch)
            

    def validation_step(self, batch_package, batch_idx):
        self.allen_ai_bart.training = False

        for name, batch in batch_package.items():
            loss = self(**batch).loss

            preds = self.model.generate(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                **self.generation_kwargs, 
            )


            allen_ai_inputs = dict(
                tokens=dict(token_ids=batch["input_ids"], 
                mask=batch["attention_mask"],
            ))
            allen_ai_outputs = dict(tokens=dict(
                token_ids=batch["decoder_input_ids"], 
                mask=batch["decoder_input_ids"] != 1,
            ))
            allen_ai_preds = self.allen_ai_bart.forward(
                source_tokens=allen_ai_inputs,
                target_tokens=allen_ai_outputs,
            )["predictions"]

            for pred_type, preds_intance in (("reg", preds), ("allen_ai", allen_ai_preds)):
                em = EM()
                recall_accuracy = RecallAcc()
                precision_accuracy = PrecisionAcc()

                for i, (x, pred, label) in enumerate(zip(batch["input_ids"], preds_intance, batch["labels"])):
                    do_print = batch_idx == 0 and i < 5
                    cleaned = OurMetric.prepare(
                        x, self.tokenizer, pred, label, do_print=do_print
                    )

                    clean_pred  = cleaned["cleaned_preds"]
                    clean_label = cleaned["cleaned_labels"]
                    assert isinstance(clean_pred,  list), type(clean_pred ).mro()
                    assert isinstance(clean_label, list), type(clean_label).mro()

                    em                .add(clean_pred, clean_label, do_print=do_print)
                    recall_accuracy   .add(clean_pred, clean_label, do_print=do_print)
                    precision_accuracy.add(clean_pred, clean_label, do_print=do_print)

                em_acc_val =         em.compute()
                recall_acc_val =     recall_accuracy.compute()
                precision_acc_val =  precision_accuracy.compute()
                f1_ACC =             2 * precision_acc_val * recall_acc_val / (precision_acc_val + recall_acc_val)
                
                self.log(f"{name}_{pred_type}_loss",           loss,               **self.logging_conf)
                self.log(f"{name}_{pred_type}_EM",             em_acc_val,         **self.logging_conf)
                self.log(f"{name}_{pred_type}_recall_ACC",     recall_acc_val,     **self.logging_conf)
                self.log(f"{name}_{pred_type}_precision_ACC",  precision_acc_val,  **self.logging_conf)
                self.log(f"{name}_{pred_type}_f1_ACC",         f1_ACC,             **self.logging_conf)

        self.allen_ai_bart.training = True

        return loss

    def configure_optimizers(self):
        """
        See ref 
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        """
        if self.is_adamw: 
            optimizer_class = transformers.AdamW
        else:
            optimizer_class = transformers.Adam

        optimizer = optimizer_class(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )

        output = dict(optimizer=optimizer)

        if SCHEDULUER_TYPES[self.scheduler_type]:
            output["lr_scheduler"] = {}
            output["lr_scheduler"]["scheduler"] = SCHEDULUER_TYPES[self.scheduler_type](
                optimizer=optimizer, **self.scheduler_kwargs
            )
            output["lr_scheduler"]["interval"] = "epoch"
            output["frequency"] = 1

        return output


    def make_dataloader(self, ds, batch_size):
        return torch.utils.data.DataLoader(
            ds, 
            collate_fn=transformers.data.data_collator.DataCollatorForSeq2Seq(
                self.tokenizer, model=self.model, padding=True
            ), 
            batch_size=batch_size, 
            num_workers=self.num_workers_dl,
            shuffle=True,
        ) 

    def train_dataloader(self):
        return self.make_dataloader(self.train_ds, self.batch_size)

    def val_dataloader(self):
        return pl.trainer.supporters.CombinedLoader(
            dict(
                eval=self.make_dataloader(
                    self.eval_ds,
                    self.eval_batch_size
                ), 
                gen=self.make_dataloader(
                    self.eval_ds,
                    self.eval_batch_size
                ), 
            ),
            "max_size_cycle",
        )


###############################################################################
# Config.
###############################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CAN CHANGE.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.
RUN_NAME = "WITH FORCE BOS 0, WD NO LABEL SMOOTHING"

RANDOM_SEED = 42

#####
## Logging / Eval frequency
#####
VAL_CHECK_INTERVAL = None
CHECK_VAL_EVERY_N_EPOCH = 5
LOG_EVERY_N_STEPS = 1
LIMIT_VAL_BATCHES = 5

SCHEDULUER_TYPES = dict(
    no_scheduler=None,
    linear=torch.optim.lr_scheduler.LinearLR,
)

@beartype
@dataclass
class Config:
    use_label_smoothing: bool = False
    learning_rate: float =      (1e-4) 
    is_adamw: bool =            True
    weight_decay: float =       0.01 
    scheduler_type: str =       "no_scheduler"
    scheduler_kwargs: dict =    field(default_factory=lambda : dict())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHOULD NEVER CHANGE.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
######
## Batch size logic. The effective batch size (512) shouldn't change.
######
NUM_GPUS = torch.cuda.device_count()
NUM_TOTAL_BATCH_SEEN = 512
TRAIN_BATCH_SIZE = 64 * 2
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE
ACCUMULATE_GRAD_BATCHES = round(NUM_GPUS * NUM_TOTAL_BATCH_SEEN / TRAIN_BATCH_SIZE)
assert NUM_TOTAL_BATCH_SEEN % TRAIN_BATCH_SIZE == 0, (
    f"{NUM_TOTAL_BATCH_SEEN % TRAIN_BATCH_SIZE} != 0"
)
assert TRAIN_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES == NUM_TOTAL_BATCH_SEEN, (
    TRAIN_BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, NUM_TOTAL_BATCH_SEEN
)

######
## Generation Kwargs
######
GENERATION_KWARGS = dict(
    num_beams=4,
    max_length=500,
)

######
## Dataset information
######
TRAIN_PATH = "./train.tsv"
TRAIN_URL = "https://github.com/najoungkim/COGS/blob/main/data/train.tsv?raw=true"
TRAIN_MD5 = "063d79fdfcacf8b04c64d430f7da6717"

EVAL_PATH = "./dev.tsv"
EVAL_URL = "https://raw.githubusercontent.com/najoungkim/COGS/main/data/dev.tsv"
EVAL_MD5 = "69ab5bf9425339f24a732785a0982744"

GEN_PATH = "./gen.tsv"
GEN_URL = "https://github.com/najoungkim/COGS/blob/main/data/gen.tsv?raw=true"
GEN_MD5 = "e6d4a859a25af9ba3319b2a27815a181"


#####
## Varia that should be changed.
#####
MODEL_NAME = "facebook/bart-base"
TRAIN_MAX_EPOCHS = 80
NUM_WORKERS_DL = get_nprocs()


###############################################################################
# Main function.
###############################################################################
def main(
    config_path: str = "./basic_config.json",
    run_name=RUN_NAME,
    wandb_entity="julesgm", 
    wandb_project="cogs_curriculum",
    ):

    # config = Config(json.loads(Path(config_path).read_text()))
    config = Config()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # These are tiny DS, probably
    maybe_download(TRAIN_PATH, TRAIN_URL, TRAIN_MD5)
    maybe_download(EVAL_PATH,  EVAL_URL,  EVAL_MD5 )
    maybe_download(GEN_PATH,   GEN_URL,   GEN_MD5  )

    model_name = MODEL_NAME
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, forced_bos_token_id=0)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    train_x, train_y = load_dataset(TRAIN_PATH)
    eval_x,  eval_y  = load_dataset(EVAL_PATH )
    gen_x,   gen_y   = load_dataset(GEN_PATH  )

    train_ds = prepare_ds(tokenizer, train_x, train_y)
    eval_ds  = prepare_ds(tokenizer, eval_x,  eval_y )
    gen_ds   = prepare_ds(tokenizer, gen_x,   gen_y  )

    pl_model = PLBart(
        model=               model, 
        tokenizer=           tokenizer, 
        train_ds=            train_ds, 
        eval_ds=             eval_ds,
        gen_ds=              gen_ds,
        train_batch_size=    TRAIN_BATCH_SIZE, 
        eval_batch_size=     EVAL_BATCH_SIZE, 
        num_workers_dl=      NUM_WORKERS_DL, 
        generation_kwargs=   GENERATION_KWARGS,
        use_label_smoothing= config.use_label_smoothing,
        learning_rate=       config.learning_rate,
        is_adamw=            config.is_adamw,
        weight_decay=        config.weight_decay,
        scheduler_type=      config.scheduler_type,
        scheduler_kwargs=    config.scheduler_kwargs,
    )

    trainer = pl.Trainer(
        max_epochs=              TRAIN_MAX_EPOCHS, 
        accelerator=             "gpu", 
        devices=                 NUM_GPUS, 
        logger=                  pl.loggers.WandbLogger(
            project=     wandb_project, 
            name=        run_name, 
            log_model=   False, 
            entity=      wandb_entity,
            config=      dict(
                **vars(config),
                num_gpus=          NUM_GPUS,
                train_batch_size=  TRAIN_BATCH_SIZE,
            )
        ),
        
        val_check_interval=      VAL_CHECK_INTERVAL,
        check_val_every_n_epoch= CHECK_VAL_EVERY_N_EPOCH,

        log_every_n_steps=       LOG_EVERY_N_STEPS,
        limit_val_batches=       LIMIT_VAL_BATCHES,
        accumulate_grad_batches= ACCUMULATE_GRAD_BATCHES,
    )
    trainer.fit(pl_model)


if __name__ == "__main__":
    fire.Fire(main)


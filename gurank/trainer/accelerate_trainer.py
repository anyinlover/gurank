from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataclasses import dataclass
from ../dataset.trec_dataset import TrecDataset, QueryTrecDataset, DataCollator, QueryDataCollator
from typing import Callable, Optional, List, Tuple
from accelerate import Accelerator, DistributedDataParallelKwargs
from tranformers import get_linear_schedule_with_warmup
import math
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate

@dataclass
class TrainerArguments:
    output_dir: str
    mdel_name_or_path: str
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_shuffle: bool = True
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 0.05
    train_epochs: int = 1,
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    save_steps: int = 1000
    save_run: bool = True
    metrics: str = "ndcg,ndcg@10"
    seed: int = 2023


class Trainer:
    """
    Trainer for training ranker model using accelerate
    """
    def __init__(
        self,
        model: nn.Module,
        args: TrainerArguments,
        train_eval_data_collator: QueryDataCollator,
        test_data_collator: DataCollator,
        train_dataset: Optional[QueryTrecDataset] = None,
        eval_dataset: Optional[QueryTrecDataset] = None,
        test_dataset: Optional[TrecDataset] = None
    ):
        self._model = model
        self._args = args
        self._train_eval_data_collator = train_eval_data_collator
        self._test_data_collator = test_data_collator
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._test_dataset = test_dataset
        self._accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
        self._is_main = self.accelerator.is_main_process

    def train(self) -> None:
        """The training entrance"""
        self.train_dataloader = DataLoader(
            self._train_dataset,
            batch_size = self._args.per_device_train_batch_size,
            collate_fn = self._train_eval_data_collator,
            suhffle = self.args.dataloader_shuffle,
            pin_memory = self.args.dataloader_pin_memory,
            num_workers = self.args.dataloader_num_workers,
        )

        self.eval_dataloader = DataLoader(
            self._eval_dataset,
            batch_size = self._args.per_device_eval_batch_size,
            collate_fn = self._train_eval_data_collator,
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

        num_training_steps = (len(self._train_dataloader) * self._args.train_epochs) // self._args.gradient_accumulation_steps

        optimizer, lr_scheduler = self._create_optimizer_and_scheduler()
        self._model, self._optimizer, self._lr_scheduler, self._train_dataloader, self._eval_dataloader = self._accelerator.prepare(
            self._model, self._optimizer, self._lr_scheduler, self._train_dataloader, self._eval_dataloader
        )
        self._training_loop()
    
    def predict(self) -> None:
        """The inference entrance"""
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size = self._args.per_device_eval_batch_size,
            collate_fn = self._test_data_collator,
            pin_memory = self.args.dataloader_pin_memory,
            num_workers = self.args.dataloader_num_workers,
        )
        self._model, self._test_dataloader = self._accelerator.prepare(self._model, self._test_dataloader)
        idxs_dfs = self._predict_loop()
        self._compute_metrics(idx_dfs)

    def _training_loop(self) -> None:
        """The training loop"""
        self.accelerator.print(f"Begin training!")
        self._model.train()
        pbar = tqdm(total=len(self._train_dataloader) * self._args.train_epochs, desc="Training Step". disable=not self._is_main)
        for epoch in range(self._args.train_epochs):
            total_loss = 0
            for step, (batch, labels) in enumerate(self.train_dataloader):
                loss, _ = self.model(**batch, labels=labels)
                loss /= self._args.gradient_accumulation_steps
                total_loss += loss.detach().float()
                self._accelerator.backward(loss)
                self._accelerator.clip_grad_norm_(self._model.parameters(), 1.0)
                overall_steps += 1
                if overall_steps % self._args.gradient_accumulation_steps == 0:
                    self._optimizer.step()
                    self._lr_scheduler.step()
                    self._optimizer.zero_grad()
                
                pbar.update()
                if overall_steps % self._args.save_steps == 0:
                    output_dir = os.path.join(self._args.output_dir, f"step_{overall_steps}")
                    self._accelerator.save_state(output_dir)
            
            self._accelerator.print(f"Finish training epoch {epoch+1}, average_loss: {total_loss / (step+1)}")
            self._evaluation_loop()
            output_dir = os.path.join(self._args.output_dir, f"epoch_{epoch+1}")
            self._accelerator.save_state(output_dir)
        
    def _evaluation_loop(self):
        """The evaluation loop"""
        self._accelerator.print("Begin evaluating!")
        self._model.eval()
        total_eval_loss = 0
        for step, (batch, labels) in tqdm(enumerate(self._eval_dataloader), desc="Evaluation loop", disable=not self._is_main):
            with torch.no_grad():
                loss, _ = self._model(**batch, labels=labels)
            total_eval_loss += loss.detach().float()
        
        self._accelerator.print(f"Finish evaluating, average_eval_loss: {total_eval_loss / (step + 1)}")
        
    def _predict_loop(self) -> List[DataFrame]:
        """The predict loop"""
        idx_dfs = []
        self._accelerator.print("Begin predicting!")
        self._model.eval()
        for idxs, batch, _ in tqdm(self._test_dataloader, desc="Predict loop", disable = not self._is_main):
            with torch.no_grad():
                _, logits = self._model(**batch, labels=labels)
            idx_dfs.append(self._create_idx_df(idxs, logits))
        
        return idx_dfs


    def _create_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer_params = {"lr": self._args.learning_rate}
        param_optimizer = list(self._model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self._args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        num_training_steps = (self._args.train_epochs * len(self._train_dataloader)) // self._args.gradient_accumulation_steps
        num_warmup_steps = math.ceil(num_training_steps * self._args.warmup_ratio)
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)

        return optimizer, lr_scheduler
    
    def _create_idx_df(self, idxs: torch.Tensor, logits: torch.Tensor) -> DataFrame:
        idxs = self._accelerator.gather_for_metrics(idxs)
        logits = logits.contiguous()
        logits = self._accelerator.gather_for_metrics(logits)
        idx_df = DataFrame({"idx": idxs.cpu().numpy(), "score": logits.cpu().numpy().astype(np.float64)})
        return idx_df

    
    def _compute_metrics(self, idx_dfs: List[DataFrame]) -> None:
        idx_df = pd.concat(idx_dfs)
        ids_df = pd.DataFrame(self._test_dataloader.dataset.ids_mapping, columns=["idx", "q_id", "doc_id"])
        run_df = pd.merge(idx_df, ids_df, on="idx", how="outer").drop("idx", axis=1)
        run = Run.from_df(run_df)
        qrels = Qrels.from_dict(self._test_dataloader.dataset.qrels_dict)
        if self._args.save_run and self._is_main:
            run_file = "run.trec"
            run.save(os.path.join(self._args.output_dir, run_file))
        metrics = self._args.metrics.split(',')
        result = evaluate(qrels, run, metrics)
        self._accelerator.print(result)
        metric_file = "metric.json"
        with open(os.path.join(self._args.output_dir, metric_file), "w") as f:
            json.dump(result, f)

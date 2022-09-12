import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from gurank.models.monobert import MonoBert
from gurank.evaluation.ndcg import NDCG
from gurank.data.msmarco import msmarco

ARGS = {
    "dataset": "~/Data/huggingface/ms_marco",
    "version": "v1.1",
    "max_length": 512,
    "pretrained_model": "~/Data/huggingface/distilroberta-base",
    "output_dir": "tests/output/monobert",
    "num_train_epochs": 1,
    "learning_rate": 2e-5,
    "batch_size": 32,
    "warmup_steps": 5000,
    "weight_decay": 0.01,
    "logging_steps": 5000,
    "amp": True,
}

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonoBert(args["pretrained_model"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args["pretrained_model"])
    dataset = msmarco(args["dataset"], tokenizer, args["version"], args["max_length"])
    training_args = TrainingArguments(output_dir=args["output_dir"],
                                    num_train_epochs=args["num_train_epochs"],
                                    learning_rate=args["learning_rate"],
                                    per_device_train_batch_size=args["batch_size"],
                                    per_device_eval_batch_size=args["batch_size"],
                                    warmup_steps=args["warmup_steps"],
                                    weight_decay=args["weight_decay"],
                                    optim="adamw_torch",
                                    evaluation_strategy="steps",
                                    logging_steps=args["logging_steps"],
                                    fp16=args["amp"],
                                    report_to="none"
                                    )
    metrics = NDCG(10)
    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=metrics,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["validation"],
                    tokenizer=tokenizer)
    trainer.train()

if __name__ == "__main__":
    train(ARGS)
    

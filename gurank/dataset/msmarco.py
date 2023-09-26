from datasets import load_dataset

def msmarco(dataset, tokenizer, version="v1.1", max_length=512):

    def transform(batch):
        passage_texts = []
        queries = []
        labels = []
        for passages, query in zip(batch["passages"], batch["query"]):
            passage_texts.extend(passages["passage_text"])
            labels.extend(passages["is_selected"])
            queries.extend([query] * len(passages["is_selected"]))
        result = tokenizer(queries, passage_texts, padding=True,
                        truncation="longest_first", max_length=max_length)
        result["labels"] = labels
        return result

    dataset = load_dataset(dataset, version)
    return dataset.map(transform, remove_columns=dataset["train"].column_names, batched=True)
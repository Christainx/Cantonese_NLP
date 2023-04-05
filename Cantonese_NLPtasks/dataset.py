# import logging
# import torch
# from tqdm.auto import tqdm
import datasets
from datasets import load_dataset, DownloadMode
from transformers import AutoTokenizer, BertTokenizer, DataCollatorWithPadding, DataCollatorForTokenClassification
# from transformers import AutoModelForSequenceClassification
# from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
# from datasets import load_metric
# from transformers import AdamW
# from transformers import get_scheduler

# from transformers import AutoModelForSequenceClassification
# from transformers import AutoModelForTokenClassification
# from transformers import Trainer, TrainingArguments


class Dataset:
    train_dataloader = None
    eval_dataloader = None
    test_dataloader = None

def load_dataset_for_transformer_ner(config):
    # data loader
    if config.force_rebuild_dataset:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.FORCE_REDOWNLOAD)
    else:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    # raw_datasets = load_dataset(config.dataset_script)

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], padding=True, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Prepare for training
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["id", "tokens", "ner_tags", "token_type_ids"]
    )

    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names

    # We can then check that the result only has columns that our model will accept:
    # ['attention_mask', 'input_ids', 'label', 'token_type_ids']

    # Now that this is done, we can easily define our dataloaders:
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle = True, batch_size = config.train_batch_size, collate_fn = data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=config.test_batch_size, collate_fn=data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.test_dataloader = test_dataloader

    # from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
    # model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=13)
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["test"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    # )
    # trainer.train()

    return ds


def load_dataset_for_transformer(config):
    # data loader
    if config.force_rebuild_dataset:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.FORCE_REDOWNLOAD)
    else:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    # raw_datasets = load_dataset(config.dataset_script)

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def tokenize_function(example):
        # print('enter')
        # print(example)
        return tokenizer(example['text'], truncation=True, max_length=256)
        # return tokenizer(example['text'], truncation=True)
        # return tokenizer(example['text'])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare for training
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["text"]
    )

    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names

    # We can then check that the result only has columns that our model will accept:
    # ['attention_mask', 'input_ids', 'label', 'token_type_ids']

    # Now that this is done, we can easily define our dataloaders:
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle = True, batch_size = config.train_batch_size, collate_fn = data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=config.test_batch_size, collate_fn=data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.test_dataloader = test_dataloader

    return ds


def load_dataset_for_transformer_pairsent(config):
    # data loader
    if config.force_rebuild_dataset:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.FORCE_REDOWNLOAD)
    else:
        raw_datasets = load_dataset(config.dataset_script, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    # raw_datasets = load_dataset(config.dataset_script)

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def tokenize_function(example):
        # print('enter')
        # print(example)
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, max_length=256)
        # return tokenizer(example['text'], truncation=True)
        # return tokenizer(example['text'])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare for training
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2"]
    )

    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names

    # We can then check that the result only has columns that our model will accept:
    # ['attention_mask', 'input_ids', 'label', 'token_type_ids']

    # Now that this is done, we can easily define our dataloaders:
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle=True, batch_size=config.train_batch_size, collate_fn=data_collator
        # tokenized_datasets["train"], shuffle = True, batch_size = config.train_batch_size, collate_fn = data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=config.test_batch_size, collate_fn=data_collator
    )

    ds = Dataset()
    ds.train_dataloader = train_dataloader
    ds.test_dataloader = test_dataloader
    return ds
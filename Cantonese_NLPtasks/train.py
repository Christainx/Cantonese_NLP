import logging
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_metric
from torch.optim import AdamW
from transformers import get_scheduler

from evaluate import *

# def train_trial():
    # # SINGLE BATCH TRAINING!!!!!!
    # # To quickly check there is no mistake in the data processing, we can inspect a batch like this:
    # model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
    # # To make sure that everything will go smoothly during training, we pass our batch to this model:
    # for batch in train_dataloader:
    #     break
    # # {k: v.shape for k, v in batch.items()}
    # # model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    # outputs = model(**batch)
    # print(outputs.loss, outputs.logits.shape)
    #
    # for batch in train_dataloader:
    #     break
    # outputs = model(**batch)
    # print(outputs.loss, outputs.logits.shape)


def training_sequenceclass(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)

    optimizer = AdamW(model.parameters(), lr = config.learning_rate)

    num_epochs = config.epoch_num
    num_training_steps = num_epochs * len(config.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    i_step = 1

    model.train()
    for epoch in range(num_epochs):
    # for epoch in range(1):
        for batch in config.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # if i_step % 100 == 0:
            #     print(outputs.loss, outputs.logits.shape)
            loss = outputs.loss
            loss.backward()
            if i_step % config.traininfo_per_step == 0:
                print(float(loss))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # evaluation():
            if i_step % config.eval_per_step == 0:
                # progress_bar_eval = tqdm(range(len(eval_dataloader)))

                # evaluation(model, config)

                golds, preds = evaluation(model, config)
                fplog = open('../output/' + config.model_name.split('/')[-1] + '_' + config.dataset_script.split('/')[-1].split('.')[0] + '_' + str(i_step) + \
                             '.txt', 'w', encoding='utf-8')
                for i in range(len(golds)):
                    gold = golds[i]
                    pred = preds[i]
                    fplog.writelines(str(gold) + ',' + str(pred) + '\n')
                    fplog.flush()

            i_step += 1

        # model.save_pretrained(save_directory="tempmodel_epoch" + str(epoch))
        # model.train()


def training_tokenclass(config):
    model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=config.num_labels)

    optimizer = AdamW(model.parameters(), lr = config.learning_rate)

    num_epochs = config.epoch_num
    num_training_steps = num_epochs * len(config.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    i_step = 1

    model.train()
    for epoch in range(num_epochs):
    # for epoch in range(1):
        for batch in config.train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # if i_step % 100 == 0:
            #     print(outputs.loss, outputs.logits.shape)
            loss = outputs.loss
            loss.backward()
            if i_step % config.traininfo_per_step == 0:
                print(float(loss))

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # evaluation():
            if i_step % config.eval_per_step == 0:
                # progress_bar_eval = tqdm(range(len(eval_dataloader)))

                # evaluation(model, config)

                golds, preds = evaluation(model, config)
                fplog = open('../output/' + config.model_name.split('/')[-1] + '_' + config.dataset_script.split('/')[-1].split('.')[0] + '_' + str(i_step) + \
                             '.txt', 'w', encoding='utf-8')
                for i in range(len(golds)):
                    gold = golds[i]
                    pred = preds[i]
                    fplog.writelines(str(gold) + ',' + str(pred) + '\n')
                    fplog.flush()

            i_step += 1

        # model.save_pretrained(save_directory="tempmodel_epoch" + str(epoch))
        # model.train()
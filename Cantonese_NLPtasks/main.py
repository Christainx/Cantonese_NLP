import os
import sys
import time
import argparse

from config_exp import *
exp_config = Config()

os.environ["CUDA_VISIBLE_DEVICES"] = exp_config.use_gpu

from dataset import *
from train import *
from tool_logger import *


def init_logger():
    os.makedirs('./log', exist_ok=True)
    os.makedirs('../output', exist_ok=True)
    LOG_FP = './log/' + \
             exp_config.task + '_' + \
             exp_config.model_name.split('/')[-2] + "_" + \
             exp_config.model_name.split('/')[-1] + '_' + \
             time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + \
             '.txt'
    set_logger(LOG_FP)

    for name, value in vars(exp_config).items():
        logging.info('$$$$$ custom para {}: {}'.format(name, value))


def run_singlesent_class():
    init_logger()

    dataset = load_dataset_for_transformer(exp_config)
    exp_config.train_dataloader = dataset.train_dataloader
    exp_config.eval_dataloader = dataset.test_dataloader

    training_sequenceclass(exp_config)


def run_pairsent_class():
    init_logger()

    dataset = load_dataset_for_transformer_pairsent(exp_config)
    exp_config.train_dataloader = dataset.train_dataloader
    exp_config.eval_dataloader = dataset.test_dataloader

    training_sequenceclass(exp_config)


def run_ner():
    init_logger()

    dataset = load_dataset_for_transformer_ner(exp_config)
    exp_config.train_dataloader = dataset.train_dataloader
    exp_config.eval_dataloader = dataset.test_dataloader

    training_tokenclass(exp_config)


def run_batch():
    for task in TASKS:
        exp_config.task = task
        for model in MODELS:
            exp_config.model_name = model
            exp_config.update()
            run_pairsent_class()


def main(exp_config):
    parser = argparse.ArgumentParser(description="Run some downstream task.")
    # parser.add_argument("--cuda_device", "-d", dest="device", default="0", choices=[str(i) for i in range(8)])
    parser.add_argument("--task", "-t", dest="task", choices=["discusshk",
                                                              "lihkgv2_yue",
                                                              "openrice_competition",
                                                              "cmnli_trans",
                                                              "ciron_trans",
                                                              "afqmc_trans",
                                                              "wenyan_ner"])
    parser.add_argument("--model_path", "-m", dest="model_path")
    args, _ = parser.parse_known_args()
    
    exp_config.task = args.task
    exp_config.model_name = args.model_path

    # exp_config.model_name = DIR_ROOT_MODEL + "hfl/chinese-roberta-wwm-ext"
    # exp_config.model_name = DIR_ROOT_MODEL + "/Model_repo/1st/110000"
    # exp_config.model_name = DIR_ROOT_MODEL + "/Model_repo/2nd/60000"
    # exp_config.model_name = DIR_ROOT_MODEL + "/Model_repo/3rd/50000"
    # exp_config.model_name = DIR_ROOT_MODEL + "/Model_repo/mix/420000"

    exp_config.update()
    if exp_config.task == "afqmc" or exp_config.task == "cmnli" or exp_config.task == "cmnli_trans" or exp_config.task == "afqmc_trans":
        run_pairsent_class()
    elif exp_config.task == "wenyan_ner":
        run_ner()
    else:
        run_singlesent_class()


exp_config.best_score = -1
for i in range(exp_config.retry_num):
    if i:
      print('##### rerun the experiment: {}'.format(i))
    main(exp_config)
logging.info('##### best score: {}'.format(exp_config.best_score))

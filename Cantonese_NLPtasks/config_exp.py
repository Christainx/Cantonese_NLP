from config_global import *

MODELS = [
    # "huggingface/bert-base-chinese",
    # "hfl/chinese-bert-wwm",
    "hfl/chinese-electra-180g-base-discriminator",
    "hfl/chinese-roberta-wwm-ext",
    # "hfl/chinese-roberta-wwm-ext-large",
    # "baidu/ernie-1.0",
]

SELF_MODELS = [

]

TASKS = [
    "afqmc",
    "ciron",
    "cmnli",
    "tnews",
    # "afqmc_yue",
    # "cmnli_yue",
    "ciron_yue",
    "openrice_yue",
    "lihkgv2_yue"
]

DATASCRIPTS = [
    "dataloader_afqmc.py"
    "dataloader_ciron.py",
    "dataloader_cmnli.py",
    "dataloader_tnews.py",
    "dataloader_afqmc_yue.py",
    "dataloader_ciron_yue.py",
    "dataloader_cmnli_yue.py",
    "dataloader_openrice.py",
    "dataloader_lihkgv2.py",
]


class Config:
    def __init__(self):
        # hardware
        self.use_gpu = "0"

        self.task = None
        self.model_name = None
        self.dataset_script = None
        self.num_labels = -1

        self.train_batch_size = -1
        self.eval_batch_size = -1
        self.test_batch_size = -1
        self.eval_per_step = -1

        # training
        self.force_rebuild_dataset = False
        self.epoch_num = 3
        self.learning_rate = 2e-5
        self.retry_num = 5

    def update(self):
        if self.task == "weibo_covid":
            self.num_labels = 6
            self.train_sample_cnt = 8606

            self.train_batch_size = int(1 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 1000 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_weibo_covid.py"

        elif self.task == "afqmc" or self.task == "afqmc_yue":
            self.num_labels = 2
            self.train_sample_cnt = 8606

            self.train_batch_size = int(2 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 1000 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            if self.task == "afqmc":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_AFQMC.py"
            elif self.task == "afqmc_yue":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_AFQMC_yue.py"

        elif self.task == "afqmc_trans":
            self.num_labels = 2
            self.train_sample_cnt = 3456

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 4 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_AFQMC_trans.py"

            self.learning_rate = 2e-5

        elif self.task == "ciron" or self.task == "ciron_yue":
            self.num_labels = 5
            self.train_sample_cnt = 7015

            self.train_batch_size = int(1 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 70 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            if self.task == "ciron":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_ciron.py"
            elif self.task == "ciron_yue":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_ciron_yue.py"

        elif self.task == "ciron_trans":
            self.num_labels = 5
            self.train_sample_cnt = 700
            self.epoch = 5

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 1 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_ciron_trans.py"

        elif self.task == "cmnli" or self.task == "cmnli_yue":
            self.num_labels = 4
            self.train_sample_cnt = 391783

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 1000 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.epoch_num = 10

            if self.task == "cmnli":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_CMNLI.py"
            elif self.task == "cmnli_yue":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_CMNLI_yue.py"

        elif self.task == "cmnli_trans":
            self.num_labels = 4
            self.train_sample_cnt = 9936

            self.epoch_num = 10

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 40 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_CMNLI_trans.py"

        elif self.task == "tnews":
            self.num_labels = 15
            self.train_sample_cnt = 53360

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 250 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_tnews_public.py"

        elif self.task == "openrice_yue" or self.task == "openrice_competition":
            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_openrice.py"
            self.num_labels = 5
            self.train_sample_cnt = 53999

            self.train_batch_size = int(2 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 100 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

            if self.task == "openrice_yue":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_openrice.py"
            elif self.task == "openrice_competition":
                self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_openrice_competition.py"

        elif self.task == "lihkgv2_yue":
            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_lihkgv2.py"
            self.num_labels = 20
            self.train_sample_cnt = 10000

            self.train_batch_size = int(2 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 32 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

        elif self.task == "discusshk":
            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_discusshk.py"
            self.num_labels = 10
            self.train_sample_cnt = 32000

            self.epoch = 10

            self.train_batch_size = int(2 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 80 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

        elif self.task == "wenyan_ner":
            self.dataset_script = DIR_ROOT_DATALOADER + "dataloader_wenyan_ner.py"
            self.num_labels = 13
            self.train_sample_cnt = 1902

            self.epoch_num = 10

            self.train_batch_size = int(4 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 10 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4

        else:
            self.num_labels = 5
            self.train_sample_cnt = 10000

            self.train_batch_size = int(1 * GPU_MEM)
            self.eval_batch_size = self.train_batch_size * 4
            self.test_batch_size = self.train_batch_size * 4
            self.eval_per_step = 250 * GPU_MEM
            self.traininfo_per_step = self.eval_per_step / 4



import os
from multiprocessing import Process

# config
TASK = ["cmnli_trans", "ciron_trans", "afqmc_trans", "lihkgv2_yue", "discusshk", "openrice_competition"]
# MODEL_PATH = ["/home/zgj/workspace/code/Cantonese_transformer/Model_repo/chinese-roberta-wwm-ext-parallel-finetuned_mandarin-nonreverse/"]
MODEL_PATH = [""]
CHECKPOINT = ["10000", "20000","30000", "40000", "50000", "60000"]
DEVICE = ["0", "1", "2", "3"]


def run_proc(param: list):
    for d, t, m in param:
        os.system(f"python main.py {d} -t {t} -m {m}")


if __name__ == '__main__':
    num_proc = len(DEVICE)
    total_times = len(MODEL_PATH) * len(TASK) * len(CHECKPOINT)
    params = {i: [] for i in range(num_proc)}
    i = 0
    for model in MODEL_PATH:
        for task in TASK:
            for ckpt in CHECKPOINT:
                params[i % num_proc].append((i % num_proc, task, model+ckpt))
                i += 1

    for i in range(num_proc):
        p = Process(target=run_proc, args=(params[i],))
        p.start()

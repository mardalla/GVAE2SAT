import os
import subprocess
import sys
import time

import torch

sys.path.append("..")
sys.path.append("../GWAE2SAT")

from GWAE2SAT.data import get_train_val_dataloader, get_max_shape
from GWAE2SAT.generate import generate
from GWAE2SAT.model import get_vgae
from GWAE2SAT.train import train


def run_i4vk(src, model_dir):
    pwd = subprocess.getoutput("pwd")
    os.chdir(model_dir)
    subprocess.call(["python", "../write_stats.py"])
    eval_conv = ["python", "eval/conversion.py",
                 "--src", src,
                 "-s", "dataset/test_set"]
    if model_dir.endswith("EGNN2S"):
        eval_conv += ["--data_name", "test_set"]
    else:
        if not os.path.isdir("test_set"):
            subprocess.call(["mkdir", "test_set"])
        for file in os.listdir(src):
            subprocess.call(["cp", os.path.join(src, file), "test_set"])
        
    time0 = time.time()
    subprocess.call(eval_conv)
    time_conv = time.time() - time0

    time0 = time.time()
    subprocess.call(["python", "main_train.py",
                     "--data_name", "test_set",
                     "--epoch_num", "201"])
    time_train = time.time() - time0

    time0 = time.time()
    for repeat in range(10):
        subprocess.call(["python", "main_test.py",
                         "--data_name", "test_set",
                         "--epoch_load", "200",
                         "--repeat", f"{repeat}"])
    time_gen = time.time() - time0

    with open("time_log.txt", "w") as file:
        file.writelines([
            f"Preprocessing: {time_conv}\n",
            f"Training: {time_train}\n",
            f"Generation: {time_gen}\n"])

    time0 = time.time()
    for repeat in range(10):
        if model_dir.endswith("EGNN2S"):
            graph_path = f"graphs/test_set_EGNN_3_32_preTrue_dropFalse_yield1_0_200_{repeat}"
        else:
            graph_path = f"graphs/test_set_SAGE_3_32_preTrue_dropFalse_yield1_0_200_{repeat}"
            subprocess.call(["rm", "-r", "test_set"])
        if not os.isdir("graph_path"):
            graph_path = f"graphs/test_set_SAGE_3_32_preTrue_dropFalse_yield1_0_150_{repeat}"
        for dat in os.listdir(graph_path):
            eval_conv = ["python", "eval/conversion.py",
                         "--src", os.path.join(graph_path, dat),
                         "--store-dir", "formulas/",
                         "--repeat", f"{repeat}"]
            if model_dir.endswith("EGNN2S"):
                eval_conv += ["--data_name", "test_set",
                              "--action=fg2sat"]
            else:
                eval_conv += ["--action=lcg2sat"]
            subprocess.call(eval_conv)

    time_post = time.time() - time0
    with open("time_log.txt", "a") as file:
        file.writelines([
            f"Postprocessing: {time_post}\n"])
        
    os.chdir(pwd)


def run_W2SAT(src):
    pwd = subprocess.getoutput("pwd")
    os.chdir("/home/dcrowley/w2sat/w2sat/")
    subprocess.call(["rm", "dataset/formulas/*"])
    subprocess.call(["cp",
                     os.path.join(src, "*.cnf"), "dataset/formulas/"])
    
    if not os.path.isdir("model/embeddings"):
        if not os.path.isdir("model"):
            os.mkdir("model")
        os.mkdir("model/embeddings")
    else:
        subprocess.call(["rm", "model/embeddings/*"])
        
    time0 = time.time()
    subprocess.call(["python", "paralleEmbedding.py"])
    time_emb = time.time() - time0
    time0 = time.time()
    subprocess.call(["python", "paralleGeneration.py"])
    time_gen = time.time() - time0

    with open("time_log.txt", "w") as file:
        file.writelines([
            f"Embedding: {time_emb}\n",
            f"Generation: {time_gen}\n"])

    for file in os.listdir("dataset/formulas"):
        subprocess.call(["rm",
                         os.path.join("dataset/formulas",
                                      file)])    
    os.chdir(pwd)


def run_GWAE2SAT(gen, train_set, test_nis_train):
    max_shape = get_max_shape(train_set)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(1024)

    time0 = time.time()
    train_dl, val_dl = get_train_val_dataloader(train_set, 16, max_shape, .1)
    time_dl = time.time() - time0

    time0 = time.time()
    vgae = get_vgae(max_shape, 32, device,
                    128, 256, 2, 3, .4)
    optim = torch.optim.Adam(vgae.parameters(), lr=.0003, weight_decay=.1)
    mae_loss = torch.nn.L1Loss()

    state_dict = train(vgae, train_dl, val_dl, 200, 16,
                       optim, mae_loss, 9, .01,
                       f"/home/dcrowley/GWAE2SAT/{test_nis_train}_models/", device)

    time_train = time.time() - time0

    time0 = time.time()
    generate(state_dict, gen,
             f"/home/dcrowley/GWAE2SAT/{test_nis_train}_generations", 16, device,
             "clause", 1, 128, 256, 2, 3, .4, 10, polarity_ratio=True)
    time_gen = time.time() - time0

    with open(f"/home/dcrowley/GWAE2SAT/{test_nis_train}_time_log.txt", "w") as file:
        file.writelines([
            f"Preprocessing: {time_dl}\n",
            f"Training: {time_train}\n",
            f"Generation: {time_gen}\n"])

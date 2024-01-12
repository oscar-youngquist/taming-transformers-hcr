from taming.models.vqgan import VQModel
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from taming.data.utils import custom_collate
import matplotlib.pyplot as plt

base_config_path = ["configs/custom_vqgan_eval.yaml"]

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

print("building model")

configs = [OmegaConf.load(cfg) for cfg in base_config_path]
config = OmegaConf.merge(*configs)

model = instantiate_from_config(config.model)

print("Loading datasets")

# load the data of interest
data = instantiate_from_config(config.data)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()

test_data_loader = data.val_dataloader()
train_data_loader = data.train_dataloader()

print(test_data_loader.__len__())
print(train_data_loader.__len__())

# set the models to eval mode
model.eval()
model.loss.discriminator.eval()

model.cuda()
# make a directory to hold qualtative eval images
output_path = "/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/taming-transformers/vqgan_eval/DeepAccident2/negatives/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

batch_num = 3000

# iterate over the test set
for batch in test_data_loader:
    print(batch_num)
    # don't track the gradient
    with torch.no_grad():
        log_dict = model.log_images(batch)
    # plt.imshow(batch["image"][2].numpy())

    # print(log_dict["dis_diff"])
    # print(log_dict["feature_diff"].size())
    
    log_dict["inputs"] = log_dict["inputs"].permute(0,2,3,1)
    log_dict["reconstructions"] = log_dict["reconstructions"].permute(0,2,3,1)
    log_dict["feature_diff"] = log_dict["feature_diff"].permute(0,2,3,1)
    

    log_dict["inputs"] = log_dict["inputs"].cpu()
    log_dict["reconstructions"] = log_dict["reconstructions"].cpu()
    log_dict["feature_diff"] = log_dict["feature_diff"].cpu()
    
    # print(log_dict["feature_diff"].size())
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True)
    
    axes[0].imshow(log_dict["inputs"][0].numpy())
    axes[0].axis('off')
    axes[0].set_title("Input")
    axes[1].imshow(log_dict["reconstructions"][0].numpy())
    axes[1].axis('off')
    axes[1].set_title("Recon.")
    axes[2].imshow(log_dict["feature_diff"][0].numpy())
    axes[2].axis('off')
    axes[2].set_title("Diff.")
    new_path = os.path.join(output_path, "neg_eval_sample_{:d}.png".format(batch_num))
    batch_num += 1
    fig.savefig(new_path)
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True)
    
    axes[0].imshow(log_dict["inputs"][1].numpy())
    axes[0].axis('off')
    axes[0].set_title("Input")
    axes[1].imshow(log_dict["reconstructions"][1].numpy())
    axes[1].axis('off')
    axes[1].set_title("Recon.")
    axes[2].imshow(log_dict["feature_diff"][1].numpy())
    axes[2].axis('off')
    axes[2].set_title("Diff.")
    new_path = os.path.join(output_path, "neg_eval_sample_{:d}.png".format(batch_num))
    batch_num += 1
    fig.savefig(new_path)
    plt.close()


    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=True)
    
    axes[0].imshow(log_dict["inputs"][2].numpy())
    axes[0].axis('off')
    axes[0].set_title("Input")
    axes[1].imshow(log_dict["reconstructions"][2].numpy())
    axes[1].axis('off')
    axes[1].set_title("Recon.")
    axes[2].imshow(log_dict["feature_diff"][2].numpy())
    axes[2].axis('off')
    axes[2].set_title("Diff.")
    new_path = os.path.join(output_path, "neg_eval_sample_{:d}.png".format(batch_num))
    batch_num += 1
    fig.savefig(new_path)
    plt.close()
    # plt.show()

    if (batch_num >= 5000):
        break
import numpy as np
import moxing as mox   # 创建解析
from torch.utils.data import DataLoader, Dataset
import torch


import argparse
# 创建解析
parser = argparse.ArgumentParser(description='train tang')

# 添加参数
parser.add_argument('--data_url', type=str, default="./data/tang.npz", help='path where the dataset is saved')
parser.add_argument('--train_url', type=str, default="./train", help='path where the model is saved')

# 解析参数
args = parser.parse_args()



# 输入修改
tang_file = np.load(args.data_url, allow_pickle=True)


# 输出修改
torch.save(my_net, args.train_url + "model.h5")
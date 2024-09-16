import numpy as np 
import random
import torch
import datetime
import time
import pytz


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def array_mean_std(numbers):
    array = np.array(numbers)
    return np.round(np.mean(array), 3), np.round(np.std(array), 3)

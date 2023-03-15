import os
import random

import numpy as np
import torch


def set_seed(seed=1):
    """

    :param seed: seed的数值可以随意设置，不清楚有没有推荐数值
    :return: None
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    # torch.manual_seed(seed)应该已经为所有设备设置seed，但是torch.cuda.manual_seed(seed)在没有gpu时也可调用
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
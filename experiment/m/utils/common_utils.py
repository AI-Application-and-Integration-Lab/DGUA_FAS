#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import torch
import numpy as np
import os
from typing import Union, Dict, Optional, Tuple
from torch import Tensor
from sys import platform

from utils import logger
from utils.ddp_utils import is_master
from cvnets.layers import norm_layers_tuple


def check_compatibility():
    ver = torch.__version__.split(".")
    major_version = int(ver[0])
    minor_version = int(ver[0])

    if major_version < 1 and minor_version < 7:
        logger.error(
            "Min pytorch version required is 1.7.0. Got: {}".format(".".join(ver))
        )


def check_frozen_norm_layer(model: torch.nn.Module) -> (bool, int):

    if hasattr(model, "module"):
        model = model.module

    count_norm = 0
    frozen_state = False
    for m in model.modules():
        if isinstance(m, norm_layers_tuple):
            frozen_state = m.weight.requires_grad

    return frozen_state, count_norm


def device_setup(opts):
    random_seed = getattr(opts, "common.seed", 0)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    is_master_node = is_master(opts)
    if is_master_node:
        logger.log("Random seeds are set to {}".format(random_seed))
        logger.log("Using PyTorch version {}".format(torch.__version__))

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        if is_master_node:
            logger.warning("No GPUs available. Using CPU")
        device = torch.device("cpu")
        n_gpus = 0
    else:
        if is_master_node:
            logger.log("Available GPUs: {}".format(n_gpus))
        device = torch.device("cuda")

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn

            torch.backends.cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            if is_master_node:
                logger.log("CUDNN is enabled")

    setattr(opts, "dev.device", device)
    setattr(opts, "dev.num_gpus", n_gpus)

    return opts


def create_directories(dir_path: str, is_master_node: bool) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        if is_master_node:
            logger.log("Directory created at: {}".format(dir_path))
    else:
        if is_master_node:
            logger.log("Directory exists at: {}".format(dir_path))


def move_to_device(
    opts,
    x: Union[Dict, Tensor],
    device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    *args,
    **kwargs
) -> Union[Dict, Tensor]:

    # if getattr(opts, "dataset.decode_data_on_gpu", False):
    #    # data is already on GPU
    #    return x

    if isinstance(x, Dict):
        # return the tensor because if its already on device
        if "on_gpu" in x and x["on_gpu"]:
            return x

        for k, v in x.items():
            if isinstance(v, Dict):
                x[k] = move_to_device(opts=opts, x=v, device=device)
            elif isinstance(v, Tensor):
                x[k] = v.to(device=device, non_blocking=non_blocking)

    elif isinstance(x, Tensor):
        x = x.to(device=device, non_blocking=non_blocking)
    else:
        logger.error(
            "Inputs of type  Tensor or Dict of Tensors are only supported right now"
        )
    return x


def is_coreml_conversion(opts) -> bool:
    coreml_convert = getattr(opts, "common.enable_coreml_compatible_module", False)
    if coreml_convert:
        return True
    return False

import argparse
from options.utils import load_config_file
from cvnets import arguments_model, arguments_nn_layers
from typing import Optional

def get_training_arguments(parse_args=True, config_path=None):
    parser = argparse.ArgumentParser()
    parser = arguments_nn_layers(parser=parser)
    parser = arguments_model(parser=parser)
    parser.add_argument('--common.config-file', type=str, default='./../../configs/mobilevit_xs.yaml')
    parser.add_argument('--dataset.category', type=str, default='classification')
    if parse_args:
        if config_path:
            opts = parser.parse_args(['--common.config-file', config_path])
        else:
            opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser

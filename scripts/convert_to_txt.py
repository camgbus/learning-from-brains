""""
Converts the .npy files generated from testing into .csv file

""""

import os
import argparse
from typing import Dict
import json
from datetime import datetime
from numpy import random
import pandas as pd
import math
import numpy as np
from torch import manual_seed
import sys
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))
from src.batcher import make_batcher
from src.decoder import make_decoder
from src.embedder import make_embedder
from src.trainer import make_trainer, Trainer
from src.unembedder import make_unembedder
from src.model import Model
from src import tools

def convert(config: Dict=None) -> Trainer:
    file_name = "/Users/raymond/desktop/raymond/stanford/senior/cns_research/learning-from-brains-master/results/models/downstream/ds002105/testing_results/GPT_lrs-4_hds-12_embd-768_train-decoding_lr-0001_bs-64_drp-01_2024-02-12_01-17-23/test_predictions.npy"
    data = np.load(file_name)

    save_location = "/Users/raymond/desktop/raymond/stanford/senior/cns_research/learning-from-brains-master/results/models/downstream/ds002105/testing_results/GPT_lrs-4_hds-12_embd-768_train-decoding_lr-0001_bs-64_drp-01_2024-02-12_01-17-23/test_predictions.csv"
    np.savetxt(save_location, data, delimiter=' ')


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )

    # Data pipeline settings:
    parser.add_argument(
        '--file',
        default= None,
        type=str,
        help='Convert npy to txt file. '

    )
    parser.add_argument(
        '--save-location',
        default= None,
        type=str,
        help='location to save converted file'
    )
        
   
    return parser


if __name__ == '__main__':

    convert()
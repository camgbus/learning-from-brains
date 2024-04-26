"""
Converts the .npy files generated from testing into .csv file
"""

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def convert(config: Dict=None) -> Trainer:

    overall_data  = []

    x = 30

    for i in range(x):
        file_name = "/Users/raymond/desktop/raymond/stanford/senior/cns_research/learning-from-brains-master/data/downstream/converted_data/original_data_{}.csv".format(i)
        data = np.loadtxt(file_name)
        data = data.flatten()
        overall_data.append(data)
    
    df = pd.DataFrame(overall_data)
    df = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    print(principalDf)
    x1 = np.array(principalDf['principal component 1'])
    y1 = np.array(principalDf['principal component 2'])

    overall_data  = []

    for i in range(x):
        file_name = "/Users/raymond/desktop/raymond/stanford/senior/cns_research/learning-from-brains-master/data/downstream/converted_data/ray_data_{}.csv".format(i)
        data = np.loadtxt(file_name)
        data = data.flatten()
        overall_data.append(data)
    
    df = pd.DataFrame(overall_data)
    df = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    print(principalDf)
    x2 = np.array(principalDf['principal component 1'])
    y2 = np.array(principalDf['principal component 2'])


    
    colors = ["red", "orange", "yellow", "green", "blue", "teal", "purple", "pink", "black", "gray", "brown", "slateblue", "violet", "orangered", "coral", "palegreen", "lightgray", "darkred", "salmon", "gold", "yellowgreen", "aquamarine", "cyan", "navy", "indigo", "fuchsia", "hotpink", "goldenrod", "olive", "mediumspringgreen"]

    for i in range(x):
        plt.scatter(x1[i], y1[i], s = 40, color = colors[i%30])
        plt.scatter(x2[i], y2[i], s = 40, color = colors[i%30], marker = "^")
    plt.show()
    

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
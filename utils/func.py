import re
import json
import math
import torch
import numpy as np
from tqdm.auto import tqdm


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True) # only difference


def read_jsonl(file, num=None):
    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            data.append(json.loads(line))
            
            if num and i == num:
                break

    return data

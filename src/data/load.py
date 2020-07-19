import os
from typing import Tuple

import pickle
import numpy as np

Arrays = Tuple[np.ndarray]

def load_data() -> Arrays:
    fpath = os.path.join('data','celeb.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    
    train_data, test_data = data['train'], data['test']
    return train_data, test_data
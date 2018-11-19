# loaders.py

import torch
import numpy as np
import h5py
from PIL import Image


def loader_image(path):
    return Image.open(path).convert('RGB')


def loader_torch(path):
    return torch.load(path)


def loader_numpy(path):
    return np.load(path)


def loader_h5py(path):
    with h5py.File(path, 'r') as hf:
        keys = hf.keys()
        composition = np.asarray(hf.get('composition'))
        c110 = np.array(hf.get('c110'))
        c120 = np.array(hf.get('c120'))
        c440 = np.array(hf.get('c440'))
        c111 = np.array(hf.get('c111'))
        c121 = np.array(hf.get('c121'))
        c441 = np.array(hf.get('c441'))
        xs = np.array(hf.get('xs'))
        ys = np.array(hf.get('ys'))
        time = np.array(hf.get('time'))

        data = {"c110": c110, "c120": c120, "c440": c440, "c111": c111, "c121": c121, "c441": c441, "composition": composition[:], "xs": xs, "ys": ys, "time": time}
    return data


def loader_h5pymeta(path):
    with h5py.File(path, 'r') as hf:
        seed = np.asarray(hf.get('seed'))
        #gbtt = np.asarray(hf.get('gbtt'))
        #orien = np.asarray(hf.get('orien'))
        composition = np.asarray(hf.get('composition'))
        runid = np.asarray(hf.get('run_id'))
    #data = {"seed": seed, "gbtt": int(gbtt), "orien": orien, 'runid': runid}
    data = {"seed": seed, "composition": composition, "runid": runid}
    return data

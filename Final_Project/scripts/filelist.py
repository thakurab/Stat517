# filelist.py

import math
import utils as utils
import torch.utils.data as data
from sklearn.utils import shuffle
import datasets.loaders as loaders
from pathlib import Path
import re
import numpy as np
import torch


def getgbtt(path):
    path = Path(path).name
    gbtt = int(re.findall("gbtt_(\d+)", str(path))[0])
    return gbtt


class FileList(data.Dataset):
    def __init__(
        self,
        trainfile,
        testfile,
        split_train=1.0,
        split_test=0.0,
        train=True,
        transform_train=None,
        transform_test=None,
        loader_train=loaders.loader_numpy,
        loader_test=loaders.loader_numpy
    ):

        # TODO: Split_train is not currently used. Either make use of it or
        # delete it.
        self.trainfile = trainfile
        self.testfile = testfile
        self.train = train
        self.split_test = split_test
        self.split_train = split_train
        self.transform_test = transform_test
        self.transform_train = transform_train

        self.loader_train = loader_train
        self.loader_test = loader_test

        self.preload = False

        if loader_train == 'image':
            self.loader_train = loaders.loader_image
        if loader_train == 'torch':
            self.loader_train = loaders.loader_torch
        if loader_train == 'numpy':
            self.loader_train = loaders.loader_numpy
        if loader_train == 'h5py':
            self.loader_train = loaders.loader_h5py

        if loader_test == 'image':
            self.loader_test = loaders.loader_image
        if loader_test == 'torch':
            self.loader_test = loaders.loader_torch
        if loader_test == 'numpy':
            self.loader_test = loaders.loader_numpy
        if loader_test == 'h5py':
            self.loader_test = loaders.loader_h5py

        if trainfile is not None:
            trainlist = utils.readtextfile(trainfile)
            trainlist = [x.rstrip('\n') for x in trainlist]
        else:
            trainlist = []

        if testfile is not None:
            testlist = utils.readtextfile(testfile)
            testlist = [x.rstrip('\n') for x in testlist]
        else:
            testlist = []

        if len(trainlist) == len(testlist):
            shuffle(testlist, trainlist)

        if len(trainlist) > 0 and len(testlist) == 0:
            shuffle(trainlist)

        if len(testlist) > 0 and len(trainlist) == 0:
            shuffle(testlist)

        if (self.split_train < 1.0) & (self.split_test > 0.0):
            if len(trainlist) > 0:
                num = math.floor(self.split_train*len(trainlist))
                self.image_train = trainlist[0:num]
            if len(testlist) > 0:
                num = math.floor(self.split_train*len(testlist))
                self.image_test = testlist[num+1:len(testlist)]

        elif self.split_train == 1.0:
            if len(trainlist) > 0:
                self.image_train = trainlist
            else:
                raise NotImplementedError("Data not found")

        elif self.split_test == 1.0:
            if len(testlist) > 0:
                self.image_test = testlist

        if self.preload is True:
            self.image_train_data = []
            for i in self.image_train:
                self.image_train_data.append(self.loader_train(i))

    def __len__(self):
        if self.train is True:
            return len(self.image_train)
        if self.train is False:
            return len(self.ferro_test)

    def __getitem__(self, index):
        data = {}
        #z = []
        # TODO: self.train is not currently used. Either use it or delete it.
        # It is necessary for the also unused split_train feature.
        if self.train is True:
            if len(self.image_train) > 0:
                if self.preload is True:
                    data  = self.image_train_data[index]
                else:
                    path = self.image_train[index]
                    data = self.loader_train(path)

                # Here i am loading as well as scaling my datadata
                
                c110 = (2 * ((data['c110'] - 801) / (1200 - 801))) - 1
                c120 = (2 * ((data['c120'] - 401) / (800 - 401))) - 1
                c440 = (2 * ((data['c440'] - 0.5) / (399.5 - 0.5))) - 1
                c111 = data['c111']
                c121 = data['c121']
                c441 = data['c441']
                xs = data['xs']
                ys = data['ys']
                time = (2 * ((data['time'] - 0.2) / (10000 - 0.2))) - 1

                composition = data['composition']
                composition = np.expand_dims(composition, 0)
                metadata = [c110, c120, c440, c111, c121, c441, xs, ys, time]            
                print (metadata)
                z = np.asarray(metadata)
                z = torch.from_numpy(z).unsqueeze(1).unsqueeze(2)
            
        return composition, z

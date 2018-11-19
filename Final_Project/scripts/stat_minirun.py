import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from pathlib import Path

import torch
from model import Model
import argparse
from dataloader import Dataloader
from checkpoints import Checkpoints
from train import Trainer
from pathlib import Path
import utils
import plugins
from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Minirun():
    def __init__(self, nrun=-1):
        self.args = Namespace(
            cuda=True,
            ndf=8,
            nef=8,
            wkld=0.01,
            gbweight=100,
            nlatent=9, 
            nechannels=6,
            ngchannels=1,
            resume="",
            save="mini-save",
            loader_train="h5py",
            loader_test="h5py",
            dataset_test=None,
            dataset_train="filelist",
            split_test=0.0,
            split_train=1.0,
            filename_test="./data/data.txt",
            filename_train="./data/data.txt",
            batch_size=64,
            resolution_high=512,
            resolution_wide=512,
            nthreads=32,
            images="mini-save/images",
            pre_name="save",
        )
        latest_save = sorted(list(Path("results").iterdir()))[nrun]
        self.rundate = latest_save.name
        latest_save = latest_save.joinpath("Save")
        latest_save = {"netG": latest_save}
        self.args.resume = latest_save
        checkpoints = Checkpoints(self.args)

        # Create model
        models = Model(self.args)
        self.model, self.criterion = models.setup(checkpoints)

        # Data loading
        self.dataloader = Dataloader(self.args)
        self.loader = self.dataloader.create(flag="Test")
        print("\t\tBatches:\t", len(self.loader))

        self.resolution_high = self.args.resolution_high
        self.resolution_wide = self.args.resolution_wide
        self.batch_size = self.args.batch_size
        self.ngchannels = self.args.ngchannels
        self.nechannels = self.args.nechannels
        self.nlatent = self.args.nlatent
        self.composition = torch.FloatTensor(self.batch_size, self.ngchannels, self.resolution_high, self.resolution_wide)
        self.metadata = torch.FloatTensor(self.batch_size, self.nechannels, self.resolution_high, self.resolution_wide)

        if self.args.cuda:
            self.composition = self.composition.cuda()
            self.metadata = self.metadata.cuda()

        self.composition = Variable(self.composition)
        self.metadata = Variable(self.metadata)

        self.imgio = plugins.ImageIO(self.args.images, self.args.pre_name)

    def date(self):
        return self.rundate

    def getmetadata(self):
        args = self.args
        #args.dataset_train = 'metalist'
        args.dataset_train = 'metadata'
        args.loader_train = 'h5pymeta'
        dataloader = Dataloader(args)
        loader = dataloader.create(flag="Test")
        data_iter = iter(loader)
        i = 0
        input_metadata = []
        while i < len(loader):
            i += 1
            composition, metadata = data_iter.next()
            input_composition.append(data["composition"])
            input_metadata.append(data["metadata"])
        return input_metadata

    def mini(self, return_data=True):
        data_iter = iter(self.loader)
        self.model["netG"].eval()
        i = 0
        generated_data = []
        input_data = []
        mydata = []
        while i < len(self.loader):
            i += 1
            composition, metadata = data_iter.next()
            for aa in metadata:
                mydata.append(aa)
            with open('minirun_metadata.txt', "w") as f:
                f.write(str(mydata[42]))
                #h1 = [1, 2, 3, 4]
                #for index in h1:
                #    f.write(str(mydata[index-1]))
            composition = composition.float()
            batch_size = composition.size(0)
            self.composition.data.resize_(composition.size()).copy_(composition)
            self.metadata.data.resize_(metadata.size()).copy_(metadata)

            self.model["netG"].zero_grad()

            # run the actual models
            output = self.model["netG"].forward(self.metadata)
            batch_gen = output.data.cpu()
            generated_data.append(batch_gen)
            input_data.append(composition)
        gen_data = torch.cat(generated_data, 0)
        input_data = torch.cat(input_data, 0)
        #print (gen_data.size())
        #print (input_data.size())
        
        #ypred = np.array(gen_data[0])
        #ytest = np.array(input_data[0])
        #ypred = pd.DataFrame([ypred])
        #ytest = pd.DataFrame([ytest])
        print ('----------------------------------------------------')
        #lpred = []
        #for i in ypred:
        #    for j in i:
        #        for k in j:
        #            lpred.append(k)
        # 
        #lpred = np.asarray(lpred)
        #ypred = pd.DataFrame([lpred])
        #print (ypred)
        #print ('----------------------------------------------------')
        #ltest = []
        #for i in ytest:
        #    for j in i:
        #        for k in j:
        #            ltest.append(k)
        #ltest = np.asarray(ltest)
        #ytest = ltest
        #print (ytest)
        #print (ytest[0])
        
        # Area fraction calculation for generated image from machine
        values = gen_data[42]
        positive_composition = 0
        negative_composition = 0
        ymodel = []
        for values1 in np.nditer(values):
            if values1 >= 0:
                temp = 1
            else:
                temp = 0
            ymodel.append(temp)
        
        #matrix_percentage = (positive_composition * 100) / 262144
        #ppt_percentage = (negative_composition * 100) / 262144
        #print ('Area fraction analysis for machine generated data')
        #print (matrix_percentage)
        #print (ppt_percentage)

        # Area fraction calculation for input image from phase field code
        values = input_data[42]
        positive_composition = 0
        negative_composition = 0
        ytest = []

        for values1 in np.nditer(values):
            if values1 >= 0:
                temp = 1
            else:
                temp = 0
            ytest.append(temp)

        print ('****************************************************')
        print (ytest)
        print (type(ytest))
        print ('****************************************************')
        print (ymodel)
        print (type(ymodel))
        print ('****************************************************')
        
        fpr, tpr, thresholds = metrics.roc_curve(ytest, ymodel)
        plt.figure(figsize = (12, 10))
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()
        plt.savefig('roc_curve.png')

        #matrix_percentage = (positive_composition * 100) / 262144
        #ppt_percentage = (negative_composition * 100) / 262144
        #print ('Area fraction analysis for input data from phase field simulation')
        #print (matrix_percentage)
        #print (ppt_percentage)

        
        #with open('gen_data.txt', 'w') as f:
        #    f.write(str(gen_data[:1]))

        #with open('input_data.txt', 'w') as f:
        #    f.write(str(input_data[:1]))

        if return_data:
            return gen_data, input_data
        else:
            self.imgio.update({"input": input_data, "sample": gen_data})
            #self.imgio.save(0, style='magnitude')
            self.imgio.save(0)

        accuracy = round(accuracy_score(ytest, ymodel) * 100, 2)
        print (accuracy)

if __name__ == "__main__":
    mini = Minirun()
    mini.mini(False)

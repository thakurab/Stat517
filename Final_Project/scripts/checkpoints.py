# checkpoints.py

import os
from pathlib import Path
import torch

class Checkpoints:
    def __init__(self,args):
        self.dir_save = args.save
        self.dir_load = args.resume
        self.best_state = {}
        self.best_epoch = 0
        self.best_saved = False

        if os.path.isdir(self.dir_save) == False:
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            if self.dir_load == None:
                return []
            else:
                return [i for i in self.dir_load.keys()]

    def keep(self, epoch, model):
        for name, m in model.items():
            self.best_state[name] = m.state_dict()
            self.best_epoch = epoch

        self.best_saved = False
        return None

    def save(self, epoch=None, model=None, best=False):
        if model:
            #if best:
            for name, m in model.items():
                    #torch.save(m.state_dict(), '%s/%s-model_epoch_%04d.pth' % (self.dir_save, name, epoch))
                #torch.save(m, '%s/%s-model_epoch_%04d.pth' % (self.dir_save, name, epoch))
                torch.save(m.state_dict(), '%s/%s-model_epoch_%04d.pth' % (self.dir_save, name, epoch))
        else:
            if not self.best_saved:
                for name, m in self.best_state.items():
                    torch.save(m, '%s/%s-model_epoch_%04d.pth' % (self.dir_save, name, self.best_epoch))
                    self.best_saved = True


        return None

    def load(self, model):
        # Load the latest save if given directory
        if Path(self.dir_load[model]).is_dir():
            filename = sorted(list(Path(self.dir_load[model]).glob(model+'*')))[-1]
        # If given file, load the file
        elif Path(self.dir_load[model]).is_file():
            filename = self.dir_load[model]
        else:
            raise FileNotFoundError(f"Invalid resume path specified for {model}")
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            model = torch.load(str(filename))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            raise FileNotFoundError(f"Invalid resume path specified for {filename}")

        return model

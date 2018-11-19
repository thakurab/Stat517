# train.py

import torch
import torch.optim as optim
from torch.autograd import Variable

import utils
import plugins
from pathlib import Path
from datasets import loaders
import numpy as np

class Trainer():
    def __init__(self, args, model, criterion):

        self.args = args
        self.model = model
        self.criterion = criterion

        self.port = args.port
        self.dir_save = args.save

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        if args.nepochs is None:
            self.nepochs = 0
        self.nchannels = args.ngchannels
        self.nechannels = args.nechannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nlatent = args.nlatent
        self.out_steps = args.out_steps

        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = getattr(optim, args.optim_method)

        self.optimizer = {}
        self.optimizer["netG"] = self.optim_method(model["netG"].parameters(), lr=self.learning_rate)
        self.composition = torch.FloatTensor(self.batch_size, self.nchannels, self.resolution_high, self.resolution_wide)
        self.metadata = torch.FloatTensor(self.batch_size, self.nechannels, self.resolution_high, self.resolution_wide)
        if args.out_images is None:
            self.out_images = self.batch_size + 8 - (self.batch_size % 8)
        else:
            self.out_images = args.out_images

        if args.cuda:
            self.composition = self.composition.cuda()
            self.metadata = self.metadata.cuda()

        self.composition = Variable(self.composition)
        self.metadata = Variable(self.metadata)
        
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['L2']
        self.log_loss_train.register(self.params_loss_train)

        self.log_monitor_train = plugins.Monitor(smoothing=False)
        self.params_monitor_train = ['L2']
        self.log_monitor_train.register(self.params_monitor_train)

        self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['L2']
        self.log_loss_test.register(self.params_loss_test)

        self.log_monitor_test = plugins.Monitor(smoothing=False)
        self.params_monitor_test = ['L2']
        self.log_monitor_test.register(self.params_monitor_test)

        # visualize training
        self.visualizer_train = plugins.Visualizer(self.port, 'Train',
                                                   args.images, args.pre_name)
        
        self.params_visualizer_train = {'L2': {'dtype': 'scalar', 'vtype': 'plot'}}
        
        self.visualizer_train.register(self.params_visualizer_train)

        # visualize testing
        self.visualizer_test = plugins.Visualizer(self.port, 'Test',
                                                  args.images, args.pre_name)
        
        self.params_visualizer_test = {'L2': {'dtype': 'scalar', 'vtype': 'plot'}}
        
        self.visualizer_test.register(self.params_visualizer_test)
        self.imgio_test = plugins.ImageIO(args.images, args.pre_name)

        # display training progress
        self.print_train = '[%04d/%04d][%02d/%02d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + f"{item:4}" + " %8.6f "

        # display testing progress
        self.print_test = '[%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + f"{item:4}" + " %8.6f "

        self.giterations  = 1
        self.losses_train = {}
        self.losses_test  = {}
        self.mu_sigma = {}

    def learning_rate(self, epoch):
        # training schedule
        return self.lr * ((0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))

    def get_optimizer(self, epoch, optimizer):
        lr = self.learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train(self, epoch, dataloader):
        self.log_monitor_train.reset()
        data_iter = iter(dataloader)

        self.model["netG"].train()

        # TODO: @icurtis - not sure why this isn't pythonic
        i = 0
        while i < len(dataloader):
            i += 1
            composition, metadata = data_iter.next()
            composition = composition.float()
            batch_size = composition.size(0)
            self.composition.data.resize_(composition.size()).copy_(composition)
            self.metadata.data.resize_(metadata.size()).copy_(metadata)
            # reset the gradients
            self.model["netG"].zero_grad()
            # run the actual models
            output = self.model["netG"].forward(self.metadata)
            # run the criterion
            loss_l2 = self.criterion["netG"](output, self.composition)
            loss_l2.backward()
            
            # run the optimizers
            self.optimizer["netG"].step()
            
            # log metrics
            self.losses_train['L2'] = loss_l2.item()

            self.log_monitor_train.update(self.losses_train, batch_size)
            print(self.print_train % tuple([epoch, self.nepochs, i, len(dataloader)] + [self.losses_train[key] for key in self.params_monitor_train]))

        self.model["netG"].eval()
        self.model["netG"].zero_grad()

        #samples, _ = self.model["netG"].forward(z)
        #samples, _ = self.model["netG"].forward(output)

        #temp = output.data.cpu()
        #bgen = torch.cat((temp, input_pol_grain_gbtt[:, 2:3]), 1)
        #bgen = torch.cat((temp, metadata), 1)                                  #what is 'c' in this line ??

        #temp = samples.data.cpu()
        #sgen = torch.cat((temp, input_pol_grain_gbtt[:, 2:3])1, 1)
        #sgen = torch.cat((temp, self.metadata), 1)                                  #what is 'c' in this line ??

        # Actually log average values to file
        loss = self.log_monitor_train.getvalues()
        self.log_loss_train.update(loss)
        self.visualizer_train.update(loss)
        return loss

    def test(self, epoch, dataloader):
        self.log_monitor_test.reset()
        data_iter = iter(dataloader)

        # Test mode?
        self.model["netG"].eval()

        i = 0
        while i < len(dataloader)-1:

            i += 1
            composition, metadata = data_iter.next()
            composition = composition.float()
            batch_size = composition.size(0)
            self.composition.data.resize_(composition.size()).copy_(composition)
            self.metadata.data.resize_(metadata.size()).copy_(metadata)
            # reset the gradients
            self.model["netG"].zero_grad()
            # run the actual models
            output = self.model["netG"].forward(self.metadata)
            # run the criterion
            loss_l2 = self.criterion["netG"](output, self.composition)

            # log metrics
            self.losses_test['L2'] = loss_l2.item()
            self.log_monitor_test.update(self.losses_test, batch_size)
            print(self.print_test % tuple([epoch, self.nepochs, i, len(dataloader)] + [self.losses_test[key] for key in self.params_monitor_test]))

        self.model["netG"].eval()
        self.model["netG"].zero_grad()

        loss = self.log_monitor_test.getvalues()
        self.log_loss_test.update(loss)
        #if epoch % self.out_steps == 0:
            #bgen = torch.cat((output.data.cpu(), composition), 0)
            ##logvarxy = genlogvar.data.cpu()
            #bgen = torch.cat((output, self.composition), 1)
            #composition = Variable(composition).cuda()
            #self.imgio_test.update({"input": composition, "sample": bgen})
            ##epoch = epoch.cpu()                                                                 #Added this line
            ##self.imgio_test.save(epoch)
            ##composition = composition.cpu()
            #loss['Test Input Images'] = utils.mapdomains(composition.cpu())
            #loss['Test Batch Fake Images'] = utils.mapdomains(bgen)
        self.visualizer_test.update(loss)
        return self.log_monitor_test.getvalues()

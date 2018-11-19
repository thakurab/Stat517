# main.py

import torch
import random
from model import Model
from config import parser
from dataloader import Dataloader
from checkpoints import Checkpoints
from train import Trainer
import utils

# parse the arguments
args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

# initialize the checkpoint class
checkpoints = Checkpoints(args)

# Create Model
models = Model(args)
model, criterion = models.setup(checkpoints)

# Data Loading
dataloader = Dataloader(args)
loader_train, loader_test = dataloader.create()
print("\t\tBatches:\t", len(loader_train))
print("\t\tBatches (Test):\t", len(loader_test))

# The trainer handles the training loop and evaluation on validation set
trainer = Trainer(args, model, criterion)

if args.oneshot == 0:
    # start training !!!
    loss_best = [1e10 for _ in model]
    epoch = 0
    while True:
        # train for a single epoch
        loss_train = trainer.train(epoch, loader_train)
        loss_test = trainer.test(epoch, loader_test)

        if (epoch % 500 == 0):
            checkpoints.save(epoch, model)
        
        if any(test < best for test, best in zip(loss_test.values(), loss_best)):
            model_best = True
            loss_best = loss_test.values()
            best_epoch = epoch
            checkpoints.keep(epoch, model)

        if (epoch > 0) and (epoch % args.save_steps == 0):
            checkpoints.save()

        if epoch == args.nepochs:
            break
        epoch += 1
    checkpoints.save()
else:
    epoch = 0
    while True:
        loss_test = trainer.test(epoch, loader_test)
        epoch += 20

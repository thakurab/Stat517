# dataloader.py

import math

import torch
import datasets
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import utils as utils

class Dataloader:

    def __init__(self, args):
        self.args = args

        self.loader_train = args.loader_train
        self.loader_test = args.loader_test

        if args.split_test is not None:
            self.split_test = args.split_test
        if args.split_train is not None:
            self.split_train = args.split_train
        else:
            self.split_train = 1.0
            self.split_test = 0.0
        self.dataset_train = args.dataset_train
        if args.dataset_test is not None:
            self.dataset_test = args.dataset_test
        else:
            self.dataset_test = self.dataset_train
        self.resolution = (args.resolution_wide, args.resolution_high)

        self.filename_test = args.filename_test
        self.filename_train = args.filename_train
        # TODO: Put sane catch methods in place.
        if args.batch_size is not None:
            self.batch_size = args.batch_size
        else:
            raise(Exception("Batch size not set"))
        if args.nthreads is not None:
            self.nthreads = args.nthreads
        else:
            raise(Exception("Number of threads not set"))

        if self.dataset_train == 'lsun':
            self.dataset_train = datasets.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train == 'cifar10':
            self.dataset_train = datasets.CIFAR10(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train == 'filelist':
            self.dataset_train = datasets.FileList(self.filename_train, None, split_train=self.split_train,
                                                   split_test=self.split_test, train=True,
                                                   transform_train=transforms.Compose([transforms.ToTensor()]),
                                                   transform_test=transforms.Compose([
                                                       #transforms.Scale(self.resolution),
                                                       #transforms.CenterCrop(self.resolution),
                                                       #  transforms.ToTensor(),
                                                       #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ]),
                                                   loader_train=self.loader_train,
                                                   loader_test=self.loader_test,
                                                   )

        elif self.dataset_train == 'data':
            self.dataset_train = datasets.data(self.filename_train, None, split_train=self.split_train,
                                                   split_test=self.split_test, train=True,
                                                   transform_train=transforms.Compose([transforms.ToTensor()]),
                                                   transform_test=transforms.Compose([
                                                       #transforms.Scale(self.resolution),
                                                       #transforms.CenterCrop(self.resolution),
                                                       #  transforms.ToTensor(),
                                                       #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ]),
                                                   loader_train=self.loader_train,
                                                   loader_test=self.loader_test,
                                                   )


            #c110 = data['c110']
            #c120 = data['c120']
            #c440 = data['c440']
            #c111 = data['c111']
            #c121 = data['c121']
            #c441 = data['c441']
            #composition = data['composition']
            #metadata = [c110, c120, c440, c111, c121, c441]
            #z = np.append(z[:], metadata)
            #z = torch.from_numpy(z)


        elif self.dataset_train == 'metalist':
            self.dataset_train = datasets.MetaList(self.filename_train, None, split_train=self.split_train,
                                                   split_test=self.split_test, train=True,
                                                   transform_train=transforms.Compose([transforms.ToTensor()]),
                                                   transform_test=transforms.Compose([
                                                   ]),
                                                   loader_train=self.loader_train,
                                                   loader_test=self.loader_test,
                                                   )
        
        elif self.dataset_train == 'metadata':
            self.dataset_train = datasets.MetaList(self.filename_train, None, split_train=self.split_train,
                                                   split_test=self.split_test, train=True,
                                                   transform_train=transforms.Compose([transforms.ToTensor()]),
                                                   transform_test=transforms.Compose([
                                                   ]),
                                                   loader_train=self.loader_train,
                                                   loader_test=self.loader_test,
                                                   )
            
        elif self.dataset_train == 'composition':
            self.dataset_train = datasets.data(self.filename_train, None, split_train=self.split_train,
                                                  split_test=self.split_test, train=True,
                                                  transform_train=transform.Compose([transforms.ToTensor()]),
                                                  transform_test=transforms.Compose([
                                                  ]),
                                                  loader_train=self.loader_train,
                                                  loader_test=self.loader_test,
                                                  )
            #c110 = data['c110']
            #c120 = data['c120']
            #c440 = data['c440']
            #c111 = data['c111']
            #c121 = data['c121']
            #c441 = data['c441']
            #composition = data['composition']
            #metadata = [c110, c120, c440, c111, c121, c441]
            #z = np.append(z[:], metadata)
            #z = torch.from_numpy(z)


        elif self.dataset_train == 'folderlist':
            self.dataset_train = datasets.FileList(self.filename_train, None, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                transform_test=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_train=self.loader_train,
                loader_test=self.loader_test,
                )

        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test == 'lsun':
            self.dataset_val = datasets.LSUN(db_path=args.dataroot, classes=['bedroom_val'],
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test == 'cifar10':
            self.dataset_val = datasets.CIFAR10(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.resolution),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test == 'filelist':
            self.dataset_val = datasets.FileList(self.filename_test, None, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    #transforms.Scale(self.resolution),
                    #transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_train=self.loader_train,
                loader_test=self.loader_test,
                )

        elif self.dataset_test == 'testdata':
            self.dataset_val = datasets.testdata(self.filename_test, None, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    #transforms.Scale(self.resolution),
                    #transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_train=self.loader_train,
                loader_test=self.loader_test,
                )


        elif self.dataset_test == 'metalist':
            self.dataset_val = datasets.MetaList(self.filename_test, None, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    #transforms.Scale(self.resolution),
                    #transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_train=self.loader_train,
                loader_test=self.loader_test,
                )

        elif self.dataset_test == 'folderlist':
            self.dataset_test = datasets.FileList(self.filename_test, None, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    # transforms.Scale(self.resolution),
                    # transforms.CenterCrop(self.resolution),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_train=self.loader_train,
                loader_test=self.loader_test,
                )

        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None):
        print("Loading data. \tBatch size:\t", self.batch_size)
        if flag == "Train":
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads))
            return dataloader_train

        if flag == "Test":
            dataloader_test = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads))
            return dataloader_test

        if flag == None:
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads))

            dataloader_test = torch.utils.data.DataLoader(self.dataset_val, batch_size=self.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads))
            return dataloader_train, dataloader_test

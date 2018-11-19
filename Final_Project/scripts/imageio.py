# imageio.py

import os
import numpy as np
import utils
import torch
import torchvision.utils as vutils


class ImageIO:

    def __init__(self, path, prepend):
        self.num = 0
        if os.path.isdir(path) is False:
            os.makedirs(path)
        self.path = path
        self.prepend = prepend + "-"
        self.values = {}

    def update(self, modules):
        for key, val in modules.items():
            self.values[key] = val

    def save(self, epoch):
        for key, value in self.values.items():
            composition = value.clone()
            name = os.path.join(self.path, self.prepend + '%04d_%s' % (epoch, key))
            np.save(name + '.npy', composition.numpy())

            if key == "input" or key == "sample":
                #h1 = [5, 8, 9, 10]
                vutils.save_image(composition[42], name + '.png')
                #if style == 'domains':
                #    batch_images = utils.mapdomains(composition)
                #    batch_images.div_(255)
                #    vutils.save_image(batch_images, name + '.png')
                #elif style == 'channels':
                #    domains[:, 0] = torch.where(domains[:, 0] < 0, domains[:, 0].abs().div(2), domains[:, 0])
                #    domains[:, 1] = torch.where(domains[:, 1] < 0, domains[:, 1].abs().div(2), domains[:, 1])
                #    vutils.save_image(domains, name + ".png")
                #elif style == 'magnitude':
                #    batch_images = utils.getmagnitude(domains)
                #    vutils.save_image(batch_images, name + ".png")

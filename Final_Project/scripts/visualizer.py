# visualizer.py

import os
import visdom
import numpy as np
import torchvision.utils as vutils

class Visualizer:
    def __init__(self, port, title, path=None, prepend=None):

        self.keys = []
        self.values = {}
        self.viz = visdom.Visdom(port=port)
        self.iteration = 0
        self.title = title
        if path:
            self.path = path
            if os.path.isdir(path) == False:
                os.makedirs(path)
        if prepend:
            self.prepend = prepend + '-'

    def register(self, modules):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']
            self.values[key]['vtype'] = modules[key]['vtype']
            if modules[key]['vtype'] == 'plot':
                self.values[key]['value'] = []
                self.values[key]['win'] = self.viz.line(
                    X=np.array([0]),
                    Y=np.array([0]),
                    opts=dict(
                        title=self.title,
                        xlabel='Epoch',
                        ylabel=key,
                        )
                    )
            elif modules[key]['vtype'] == 'image':
                self.values[key]['value'] = None
            elif modules[key]['vtype'] == 'images':
                self.values[key]['value'] = None
            else:
                sys.exit('Data type not supported, please update the visualizer plugin and rerun !!')

    def update(self, modules):
        for key in modules:
            if self.values[key]['dtype'] == 'scalar':
                self.values[key]['value'].append(modules[key])
            elif self.values[key]['dtype'] == 'image':
                self.values[key]['value'] = modules[key]
            elif self.values[key]['dtype'] == 'images':
                self.values[key]['value'] = modules[key]
            else:
                sys.exit('Data type not supported, please update the visualizer plugin and rerun !!')

        for key in self.keys:
            if self.values[key]['vtype'] == 'plot':
                self.viz.line(
                    X=np.array([self.iteration]),
                    Y=np.array([self.values[key]['value'][-1]]),
                    win = self.values[key]['win'],
                    update='append'
                    )
            elif self.values[key]['vtype'] == 'image':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    temp[i] = temp[i] - temp[i].min()
                    temp[i] = temp[i]/temp[i].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.image(
                        temp,
                        opts=dict(title=key, caption=self.iteration)
                        )
                else:
                    self.viz.image(
                        temp,
                        opts=dict(title=key, caption=self.iteration),
                        win=self.values[key]['win']
                        )
            elif self.values[key]['vtype'] == 'images':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        temp[i][j] = temp[i][j] - temp[i][j].min()
                        temp[i][j] = temp[i][j]/temp[i][j].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.images(
                        temp,
                        opts=dict(title=key,
                        caption=self.iteration)
                        )
                else:
                    self.viz.images(
                        temp,
                        opts=dict(title=key,
                        caption=self.iteration),
                        win=self.values[key]['win']
                        )
            else:
                sys.exit('Visualization type not supported, please update the visualizer plugin and rerun !!')
        self.iteration = self.iteration + 1

    def save(self, epoch):
        for key in self.keys:
            name = os.path.join(self.path, self.prepend + '%04d_%s' % (epoch, key))
            if self.values[key]['vtype'] == 'images':
                vutils.save_image(self.values[key]['value'], name + '.png')

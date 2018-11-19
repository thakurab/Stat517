import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from pathlib import Path

logpath = Path('./results')
dirs = sorted([d for d in logpath.iterdir() if d.is_dir()])

#  dirs = [logpath.joinpath('2018-03-07_18-25-22')]
#logpath = dirs[-1]
logpath = dirs[-1]
trainpath = logpath.joinpath('Logs/TrainLogger.txt')
testpath = logpath.joinpath('Logs/TestLogger.txt')
if trainpath.exists():
    print("Log found: ", str(trainpath))
if testpath.exists():
    print("Log found: ", str(testpath))

sk = 50
traindata = np.loadtxt(str(trainpath), skiprows=1, unpack=True)
testdata = np.loadtxt(str(testpath), skiprows=1, unpack=True)
print (traindata.shape)
print (testdata.shape)


fig, ax = plt.subplots(sharex=True, figsize=(5, 8*4/9))


def autoscale_based_on(ax, lines):
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()


dims = np.arange(sk, traindata.shape[0]+sk)
print(traindata.shape)
ax.plot(dims, traindata, label='Training Loss', alpha=0.8)
ax.plot(dims, testdata, ls='--', label='Testing Loss', alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel('Total loss')
ax.legend()
ax.legend()
#  ax.set_ylim(0, 1)
#  ax[1].set_ylabel(' loss')

fig.tight_layout()
fig.savefig('{}.png'.format(logpath.name), dpi=200)
print ('Done .. !!')

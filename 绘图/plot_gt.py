'''绘画groundtruth map'''
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import glob
from utils import plot_map

INFO = {'PaviaU_gt': 'paviaU_gt',
        'Salinas_gt': 'salinas_gt',
        'KSC_gt': 'KSC_gt',
        'gf5_gt': 'gf5_gt'}

if __name__ == '__main__':
    path = glob.glob('gt\\*.mat')
    # path = [p.split('\\')[-1] for p in path]
    for p in path:
        m = loadmat(p)
        name = p.split('\\')[-1].split('.')[0]
        gt = m[INFO[name]]
        h, w = gt.shape
        map = plot_map(gt)
        plt.figure(figsize=(w/100.0, h/100.0))
        plt.imshow(map, aspect='equal')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
        plt.margins(0, 0)
        plt.savefig('gt/{}.eps'.format(name.split('_')[0]))
        plt.close()
        print('*'*5 + 'FINISH {}'.format(name) + '*'*5)

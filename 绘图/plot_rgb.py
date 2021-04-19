from utils import rgb
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import argparse
import os

INFO = {'PaviaU': {'data_key': 'paviaU',
                   'gt_key': 'paviaU_gt',
                   'b_0': 430,
                   'b_n': 860,
                   'band': 103},
        'Salinas': {'data_key': 'salinas_corrected',
                    'gt_key': 'salinas_gt',
                    'b_0': 200,
                    'b_n': 2400,
                    'band': 204},
        'KSC': {'data_key': 'KSC',
                'gt_key': 'KSC_gt',
                'b_0': 400,
                'b_n': 2500,
                'band': 176},
        'gf5': {'data_key': 'gf5',
                'gt_key': 'gf5_gt',
                'b_0': 400,
                'b_n': 2500,
                'band': 280}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘画高光谱图像的RGB图像')
    parser.add_argument('--name', type=str, default='gf5',
                        help='DATASET NAME')
    arg = parser.parse_args()
    dataset_name = arg.name
    info = INFO[dataset_name]
    m = loadmat('data/{0}/{0}.mat'.format(dataset_name))
    data = m[info['data_key']]
    # gt = gt.astype(np.int)
    data = data.astype(np.float)
    b_0 = info['b_0']
    b_n = info['b_n']
    n = info['band']
    d = (b_n - b_0) / (n - 1)
    r, g, b = rgb(b_0 ,d)
    img = data[..., (r, g, b)]
    h, w = img.shape[:2]
    data = img.reshape((h*w, -1))
    data -= np.min(data, axis=0)
    data /= (np.max(data, axis=0) + 1e-8)
    img = data.reshape((h, w, -1))
    img *= 255
    img = img.astype(np.uint8)
    plt.figure(figsize=(w / 100.0, h / 100.0))
    plt.imshow(img, aspect='equal')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    plt.margins(0, 0)
    # plt.show()
    if not os.path.exists('rgb'):
        os.mkdir('rgb')
    plt.savefig('rgb/{}_rgb.pdf'.format(dataset_name))
    print('*'*5 + 'FINISH PLOT {}'.format(dataset_name) + '*'*5)
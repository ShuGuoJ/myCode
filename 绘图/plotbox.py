r""" 绘画数据的箱型图以观察光谱的变异性 """
from scipy.io import loadmat
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
import argparse
import numpy as np
import math
import os
import configparser
# INFO = {
#     'PaviaU': {'data_key': 'paviaU',
#                'gt_key': 'paviaU_gt'},
#     'Salinas': {'data_key': 'salinas_corrected',
#                 'gt_key': 'salinas_gt'},
#     'KSC': {'data_key': 'KSC',
#             'gt_key': 'KSC_gt'},
#     'gf5': {'data_key': 'gf5',
#             'gt_key': 'gf5_gt'}
# }
ROWS = 4
BOX = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot box figure')
    parser.add_argument('--name', type=str, required=True,
                        help='DATASET NAME')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')
    # info = INFO[arg.name]
    # 数据读取
    m = loadmat('data/{0}/{0}.mat'.format(arg.name))
    # data = m[info['data_key']]
    data = m[config.get(arg.name, 'data_key')]
    m = loadmat('data/{0}/{0}_gt.mat'.format(arg.name))
    # gt = m[info['gt_key']]
    gt = m[config.get(arg.name, 'gt_key')]
    # 数据格式转换
    data, gt = data.astype(np.float), gt.astype(np.int32)
    h, w, c = data.shape
    data = data.reshape((h*w, c))
    # 归一化
    data = scale(data)
    # 拉平
    gt = gt.flatten()
    # # 非0下标
    nc = gt.max()
    # 绘图
    figs = int(math.ceil(c / BOX))
    lines = int(math.ceil(figs / ROWS))
    for i in np.unique(gt):
        _, ax = plt.subplots(nrows=lines, ncols=4, figsize=(20, 3 * lines))
        plt.suptitle('{}_class_{}.jpg'.format(arg.name, i), fontsize=15, fontproperties={'family': 'Times New Roman'})
        # _, ax = plt.subplots(nrows=lines, ncols=4)
        axes = ax.flatten()
        indices = np.nonzero(gt == i)
        data_i = data[indices]
        # axes[i - 1].boxplot(data_i)
        for j, k in enumerate(range(BOX, c, BOX)):
            labels = [str(l) for l in range(k - BOX, k)]
            axes[j].boxplot(data_i[:, k - BOX:k], showcaps=True, showfliers=False, patch_artist=True,
                              labels=labels)
        if c % BOX != 0:
            labels = [str(l) for l in range(k, c)]
            axes[j + 1].boxplot(data_i[:, k:], showcaps=True, showfliers=False, patch_artist=True,
                            labels=labels)
        # 清空剩余子图
        for k in range(figs, len(axes)):
            axes[k].axis('off')
        if not os.path.exists('boxfigure'):
            os.makedirs('boxfigure')
        plt.savefig('boxfigure/{}_class_{}.jpg'.format(arg.name, i))
        plt.close()

    print('*'*5 + 'FINISH' + '*'*5)

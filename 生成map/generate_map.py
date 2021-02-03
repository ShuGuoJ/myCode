'''根据map文件夹中的.mat文件来绘画相应的classification map'''
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
import glob
CNAME={
    'darkblue': '#00008B',
    'aquamarine': '#7FFFD4',
    'blue': '#0000FF',
    'chartreuse': '#7FFF00',
    'cornflowerblue': '#6495ED',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F'}
COLOR = list(zip(*list(CNAME.items())))[-1]
ROOT = 'map'
SAMPLE_PER_CLASS = [10, 50, 100]
SAVE_ROOT = 'ans'


# 颜色十六进制数转换为RGB
def hex2rgb(hex):
    rgb = []
    for i in range(1, len(hex), 2):
        tmp = '0x' + hex[i:i+2]
        rgb.append(eval(tmp))
    return np.array(rgb, dtype=np.uint8)


def plot_map(gt):
    map = np.zeros_like(gt, dtype=np.uint8)
    map = np.expand_dims(map, axis=-1)
    map = np.repeat(map, 3, axis=-1)
    h, w = gt.shape
    for i in range(h):
        for j in range(w):
            index = int(gt[i, j])
            map[i, j] = hex2rgb(COLOR[index])
    return map


def main(mat_path):
    m = loadmat(mat_path)
    pred = m['pred']
    model_name = mat_path.split('\\')[-1].split('.')[0]
    map = plot_map(pred)
    height, width, channels = map.shape
    # fig, ax = plt.subplots()
    plt.figure(figsize=(width / 100.0, height / 100.0))
    plt.imshow(map, aspect='equal')
    # 去除图像周围的白边

    # 如果dpi=300，那么图像大小=height*width
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    plt.margins(0, 0)

    # plt.show()
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    plt.savefig(os.path.join(SAVE_ROOT, '{}.eps'.format(model_name)))
    # plt.clf()
    plt.close()


if __name__ == '__main__':
    mat_files = glob.glob('{}/*.mat'.format(ROOT))
    for f in mat_files:
        main(f)
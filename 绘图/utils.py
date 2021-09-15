<<<<<<< HEAD
'''根据map文件夹中的.mat文件来绘画相应的classification map'''
import numpy as np
import math
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
    'darkslategray': '#2F4F4F',
    'fuchsia': '#FF00FF',
    'lightblue': '#ADD8E6'}
COLOR = list(zip(*list(CNAME.items())))[-1]
ROOT = 'map'
SAMPLE_PER_CLASS = [10, 50, 100]


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
            if index != 0:
                map[i, j] = hex2rgb(COLOR[index])
            else:
                map[i, j] = hex2rgb('#FFFFFF')
    return map

# 计算rgb波段索引
def rgb(a_0, d):
    r = int(math.floor((700-a_0)/d))
    g = int(math.floor((546.1-a_0)/d))
    b = int(math.floor((435.8-a_0)/d))
=======
'''根据map文件夹中的.mat文件来绘画相应的classification map'''
import numpy as np
import math
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
    'darkslategray': '#2F4F4F',
    'fuchsia': '#FF00FF',
    'lightblue': '#ADD8E6'}
COLOR = list(zip(*list(CNAME.items())))[-1]
ROOT = 'map'
SAMPLE_PER_CLASS = [10, 50, 100]


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
            if index != 0:
                map[i, j] = hex2rgb(COLOR[index])
            else:
                map[i, j] = hex2rgb('#FFFFFF')
    return map

# 计算rgb波段索引
def rgb(a_0, d):
    r = int(math.floor((700-a_0)/d))
    g = int(math.floor((546.1-a_0)/d))
    b = int(math.floor((435.8-a_0)/d))
>>>>>>> 2b422f73497afd1f4ee5fe52321761de168e9451
    return r, g, b
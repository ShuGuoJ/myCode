import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

'''矩阵逆时针旋转90度'''
def rotate_matrix_90(m):
    assert len(m.shape) >= 2
    h, w = m.shape[:2]
    ans = np.zeros_like(m)
    new_shape = list(range(len(m.shape)))
    new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
    ans = ans.transpose(new_shape)
    x_coor = np.arange(h).reshape((h, 1))
    y_coor= -np.arange(w).reshape((1, w)) + w - 1
    x_coor, y_coor = np.repeat(x_coor, w, 1), np.repeat(y_coor, h, 0)
    ans[y_coor, x_coor] = m
    return ans

'''矩阵水平翻转'''
def flip_from_left2right(m):
    assert len(m.shape) >= 2
    h, w = m.shape[:2]
    ans = np.zeros_like(m)
    x_coor = np.arange(h).reshape((h, 1))
    y_coor = w - np.arange(w).reshape((1, w)) - 1
    x_coor, y_coor = np.repeat(x_coor, w, 1), np.repeat(y_coor, h, 0)
    ans[x_coor, y_coor] = m
    return ans

'''矩阵垂直旋转'''
def flip_from_up2bottom(m):
    assert len(m.shape) >= 2
    h, w = m.shape[:2]
    ans = np.zeros_like(m)
    x_coor = h - np.arange(h).reshape((h, 1)) - 1
    y_coor = np.arange(w).reshape((1, w))
    x_coor, y_coor = np.repeat(x_coor, w, 1), np.repeat(y_coor, h, 0)
    ans[x_coor, y_coor] = m
    return ans

def augment(img):
    s, h, w, c = img.shape
    augment_data = np.zeros((8*s, h, w, c), dtype=np.uint8)
    for i, x in enumerate(img):
        for j in range(8):
            if j == 4: x = flip_from_left2right(x)
            augment_data[8*i+j] = x
            x = rotate_matrix_90(x)
    return augment_data

if __name__ == '__main__':
    # # 读取图片
    # img = Image.open('liuyifei.jpg')
    # img = np.array(img)
    # # 图像旋转
    # rotation_img = rotate_matrix_90(img)
    # horizontal_img = flip_from_left2right(img)
    # vertical_img = flip_from_up2bottom(img)
    # # 显示图像
    # axi = plt.subplot(1, 4, 1)
    # axi.imshow(img)
    #
    # axi = plt.subplot(1, 4, 2)
    # axi.imshow(rotation_img)
    #
    # axi = plt.subplot(1, 4, 3)
    # axi.imshow(horizontal_img)
    #
    # axi = plt.subplot(1, 4, 4)
    # axi.imshow(vertical_img)
    #
    # plt.show()

    # 读取图片
    img = Image.open('liuyifei.jpg')
    img = img.resize((375,375))
    img = np.array(img)
    # 扩展维度
    img = np.expand_dims(img, 0)
    # 获取8张图像
    imgs = augment(img)
    # 展示
    for i in range(1, 9):
        axi = plt.subplot(2, 4, i)
        axi.imshow(imgs[i-1])
    plt.show()
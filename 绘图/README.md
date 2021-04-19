# 使用手册
## 功能
* plot_gt.py 根据gt文件夹底下的.mat文件绘画出高光谱图像的ground truth map。
* plot_pred.py 根据pred文件夹底下的.mat文件绘画出高光谱图像的prediction map。
* rename.py 为pred文件夹底下的文件添加命名前缀，如`PaviaU_XX.mat`。
* visualize.py 可视化数据集中样本的分布情况，通过tsne对数据集进行降维。
* plot_rgb.py 可视化高光谱图像的伪彩色图像，并将生成的图像存放到rgb文件夹底下。

## 代码运行
* plot_gt.py `python plot_gt.py`
* plot_pred.py `python plot_pred.py`
* rename.py `python rename.py --name xx（数据集名称）`
* visualize.py `python visualize.py -- root xx（数据文件夹根目录） -- name xx（数据集名称）`
* plot_rgb.py `python plot_rgb.py --name xx（数据集名称）`
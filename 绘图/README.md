<<<<<<< HEAD
<<<<<<< HEAD
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
=======
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
>>>>>>> 2b422f73497afd1f4ee5fe52321761de168e9451
* plot_rgb.py `python plot_rgb.py --name xx（数据集名称）`
=======
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

## 新修改
通过使用外部配置文件dataInfo.ini来存储数据集的信息，然后再使用内置模块configparser来加载dataInfo.ini配置文件。这种方法能够提过数据集信息的可移植性，便于其它程序的使用。
>>>>>>> 5c2e3e8cc01dc4431ae3f49522cda46fb970e30e

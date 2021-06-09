# The utils of deep learning
## Description
The repository is used to store the basic code about deep learning and data preprocess during my postgraduate to reduce the developing time.
## Installation
1. pytorch  
2. numpy  
3. torch_geometric  
4. torch_scatter

## Instruction
1. `Trainer.py` contains a class named as `Trainer`, which can be used to train and evaluate deep learning model. It can adopt in any deep learning model by slightly modifying the code of the output of dataset, the input of model and the calculation of loss function.

2. `生成map` directory contains a `generate_map.py` which is to generate the classification map of the corresponding classification result on specific dataset. The directory of `map` contains the classfication result and the directory of `ans` contains the output of `generate_map.py`.

3. `Monitor.py` contains a class named as `GradMonitor` that is to track the parameters' gradient.

4. `旋转矩阵.py` contains three functions that is to rotate and flip matrix, not only RGB or gray image.

5. `构造图.py` contains two functions. one is `get_edge_index` that is to gain adjacent relationship according to superpixel segmentation result. The other is `get_node` of which the function is the construction of node feature.

6. `生成所有的样本配对.py` contains three function. Their function is construction of the all sample combination according one number through the binary search. It can save consumption of memory and time.

7. `visualize.py` contains a function to facilitate us to visualize sample point.

8. `Loss.py` contains a contrastive loss.
9. `绘图` directory contains plot experiment image about hyperspectral image classification tool. Different from `生成map`, it contains many finer tools to plot ground truth, full prediction map, feature visualization, and false-color image of hyperspectral image.
10. `Summary.py` contains a reimplemented summary function in torchsummary module, which can calculate the number of trainable parameters in a neural network without any other parameters just according to an instance of an abstract class.
## Contribution

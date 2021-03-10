import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score


# 计算模型的recall, AA, OA and kappa
def measure(y_pred, y_true):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = np.array(y_pred)
    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
    else:
        y_true = np.array(y_true)
    # 计算类别 recall 值
    class_recall = recall_score(y_true, y_pred, average=None)
    # 计算平均 recall
    AA = class_recall.mean()
    # 计算准确率
    OA = accuracy_score(y_true, y_pred)
    # 计算 kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    res = {'class_recall': class_recall.tolist(),
           'AA': AA,
           'OA': OA,
           'kappa': kappa}
    return res


# if __name__ == '__main__':
#     y_true = [2, 0, 2, 2, 0, 1]
#     y_pred = [0, 0, 2, 2, 0, 2]
#     print(measure(y_pred, y_true))
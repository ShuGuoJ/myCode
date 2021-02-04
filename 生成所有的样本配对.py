# 二分查找小于或等于v的值所对应的下表
def bi_search(n_sample, v):
    '''
    :param n_sample: 总样本数
    :param v: 查找的key值
    :param f: 数列的通项公式
    :return: 下标
    '''
    begin = 0
    end = n_sample - 1
    tmp = None
    while begin <= end:
        middle = (begin + end) // 2
        tmp = f(n_sample, middle)
        if tmp > v:
            end = middle - 1
        else:
            begin = middle + 1
    return tmp, begin - 1


# 计算数列前k项的和
def f(n, k):
    '''
    :param n: 样本数
    :param k:
    :return:
    '''
    return (k * ((2 * n - 1) - k)) // 2


# 主函数
def get_pair(self, index):
    v, sample_1_index = bi_search(len(self.indices), index)
    sample_2_index = index + sample_1_index + 1 - v
    return sample_1_index, sample_2_index
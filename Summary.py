'''统计神经网络模型各层参数的数量'''
from torch import nn


def summary(net: nn.Module):
    single_dotted_line = '-' * 40
    double_dotted_line = '=' * 40
    star_line = '*' * 40
    content = []
    def backward(m: nn.Module, chain: list):
        children = m.children()
        params = 0
        chain.append(m._get_name())
        try:
            child = next(children)
            params += backward(child, chain)
            for child in children:
                params += backward(child, chain)
            # print('*' * 40)
            # print('{:>25}{:>15,}'.format('->'.join(chain), params))
            # print('*' * 40)
            if content[-1] is not star_line:
                content.append(star_line)
            content.append('{:>25}{:>15,}'.format('->'.join(chain), params))
            content.append(star_line)
        except:
            for p in m.parameters():
                if p.requires_grad:
                    params += p.numel()
            # print('{:>25}{:>15,}'.format(chain[-1], params))
            content.append('{:>25}{:>15,}'.format(chain[-1], params))
        chain.pop()
        return params
    # print('-' * 40)
    # print('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    # print('=' * 40)
    content.append(single_dotted_line)
    content.append('{:>25}{:>15}'.format('Layer(type)', 'Param'))
    content.append(double_dotted_line)
    params = backward(net, [])
    # print('=' * 40)
    # print('-' * 40)
    content.pop()
    content.append(single_dotted_line)
    print('\n'.join(content))
    return params


from torchvision import models
net = models.vgg11()
summary(net)

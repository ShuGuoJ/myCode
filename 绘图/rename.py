import os
import glob
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量修改文件名')
    parser.add_argument('--name', type=str, default='PaviaU',
                        help='DATASET NAME')
    arg = parser.parse_args()
    root = 'pred'
    filelist = glob.glob(os.path.join(root, '*.pdf'))
    for f in filelist:
        fname = f.split('\\')[-1]
        os.rename(f, os.path.join(root, '{}_{}'.format(arg.name, fname)))
        print('{}>>>{}_{}'.format(fname, arg.name, fname))
    print('*'*5 + 'FINISH' + '*'*5)
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt


def read_directory(dir_name):

    to_plot = {}
    names = ['partial_n', 'pu_partial_n', 'minus_pu_partial_n',
             'nnpu', 'nnpnu', 'pn', 'sep_partial_n']

    for f in listdir(dir_name):
        if f != 'all' and isfile(join(dir_name, f)):
            pers, va_pers, losses = read_one_file(join(dir_name, f))
            print(f)
            for name in names:
                if f.startswith('mnist_{}'.format(name)):
                    if name not in to_plot:
                        to_plot[name] = [], [], []
                    to_plot[name][0].append(pers)
                    to_plot[name][1].append(va_pers)
                    to_plot[name][2].append(losses)

    plt.figure()
    plt.title('test accuracy')
    for lab in to_plot:
        m = np.mean(np.array(to_plot[lab][0]), axis=0)
        s = np.std(np.array(to_plot[lab][0]), axis=0)
        plt.plot(m, label=lab)
        plt.fill_between(np.arange(len(m)), m-s/2, m+s/2, alpha=0.5)
    plt.legend()

    plt.figure()
    plt.title('validation loss')
    for lab in to_plot:
        m = np.mean(np.array(to_plot[lab][1]), axis=0)
        s = np.std(np.array(to_plot[lab][1]), axis=0)
        plt.plot(m, label=lab)
        plt.fill_between(np.arange(len(m)), m-s/2, m+s/2, alpha=0.5)
    plt.legend()

    plt.figure()
    plt.title('training loss')
    for lab in to_plot:
        m = np.mean(np.array(to_plot[lab][2]), axis=0)
        s = np.std(np.array(to_plot[lab][2]), axis=0)
        plt.plot(m, label=lab)
        plt.fill_between(np.arange(len(m)), m-s/2, m+s/2, alpha=0.5)
    plt.legend()


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    pers = []
    va_pers = []
    losses = []
    for i, line in enumerate(content):
        if line == '\n':
            pers = []
            va_pers = []
            losses = []
        if line.startswith('pp'):
            a = line.split()
            pers.append(float(a[2]))
        if line.startswith('np'):
            a = line.split()
            pers[-1] += float(a[2])
        if line.startswith('Test set: Error:'):
            a = line.split()
            pers.append(float(a[3]))
        if line.startswith('Test set: Accuracy:'):
            pers.append(float(line[-8:-3]))
        if line.startswith('valid') and content[i+1].startswith('Epoch'):
            a = line.split()
            if len(va_pers) <= 2:
                va_pers.append(float(a[1]))
            else:
                va_pers.append(float(a[1])*1+va_pers[-1]*0)
        if line.startswith('Validation') and content[i+1].startswith('Epoch'):
            a = line.split()
            va_pers.append(float(a[2]))
        if line.startswith('Epoch'):
            a = line.split()
            losses.append(float(a[4]))
    return pers[2:], va_pers[2:], losses[2:]


read_directory(sys.argv[1])
plt.show()

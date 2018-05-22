import sys
from os import listdir
from os.path import isfile, join
# import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt


def read_directory(dir_name):
    perss = []
    va_perss = []
    lossess = []
    names = []
    for f in listdir(dir_name):
        if f != 'all' and isfile(join(dir_name, f)):
            pers, va_pers, losses = read_one_file(join(dir_name, f))
            print(f)
            # pers = scipy.ndimage.filters.gaussian_filter1d(
            #     np.array(pers), 0.5)
            names.append(f)
            perss.append(pers)
            va_perss.append(va_pers)
            lossess.append(losses)
    plt.figure()
    plt.title('test error')
    for i, pers in enumerate(perss):
        plt.plot(pers, label=names[i])
    plt.legend()
    plt.figure()
    plt.title('validation loss')
    for i, va_pers in enumerate(va_perss):
        plt.plot(va_pers, label=names[i])
    plt.legend()
    plt.figure()
    plt.title('training loss')
    for i, losses in enumerate(lossess):
        plt.plot(losses, label=names[i])
    plt.legend()


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    pers = []
    va_pers = []
    losses = []
    for i, line in enumerate(content):
        if line.startswith('pp'):
            a = line.split()
            pers.append(float(a[2]))
        if line.startswith('np'):
            a = line.split()
            pers[-1] += float(a[2])
        if line.startswith('Test set: Error:'):
            a = line.split()
            pers.append(float(a[3]))
        if line.startswith('valid') and content[i+1].startswith('Epoch'):
            a = line.split()
            if len(va_pers) <= 2:
                va_pers.append(float(a[1]))
            else:
                va_pers.append(float(a[1])*1+va_pers[-1]*0)
        if line.startswith('Epoch'):
            a = line.split()
            losses.append(float(a[4]))
    return pers[2:], va_pers[2:], losses[2:]


read_directory(sys.argv[1])
plt.show()

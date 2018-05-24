import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def read_directory(dir_name):
    w0s = []
    w1s = []
    names = []
    for f in listdir(dir_name):
        if f != 'all' and isfile(join(dir_name, f)):
            weights0, weights1 = read_one_file(join(dir_name, f))
            print(f)
            weights0 = scipy.ndimage.filters.gaussian_filter1d(
                np.array(weights0), 1)
            weights1 = scipy.ndimage.filters.gaussian_filter1d(
                np.array(weights1), 1)
            names.append(f)
            w0s.append(weights0)
            w1s.append(weights1)
    plt.figure()
    plt.title('w0')
    for i, pers in enumerate(w0s):
        plt.plot(pers, label=names[i])
    plt.legend()
    plt.figure()
    plt.title('w1')
    for i, va_pers in enumerate(w1s):
        plt.plot(va_pers, label=names[i])
    plt.legend()


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    weights0 = []
    weights1 = []
    counter = 0
    for i, line in enumerate(content):
        if line.startswith('['):
            if counter % 2 == 0:
                summ = 0
                for num in line[1:].split():
                    summ += float(num)
                if content[i+1][-2] == ']':
                    for num in content[i+1][:-2].split():
                        summ += float(num)
                else:
                    for num in content[i+1].split():
                        summ += float(num)
                    for num in content[i+2][:-2].split():
                        summ += float(num)
                avg = summ/10
                if counter % 4 == 0:
                    weights0.append(avg)
                else:
                    weights1.append(avg)
            counter += 1
    return weights0, weights1


read_directory(sys.argv[1])
plt.show()

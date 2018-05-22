import sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def read_directory(dir_name):
    pers = []
    for f in listdir(dir_name):
        if f != 'all' and isfile(join(dir_name, f)):
            pers = read_one_file(join(dir_name, f))
            print(f)
            print(pers)
            plt.plot(pers, label=f)
    return pers


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    pers = []
    for line in content:
        if line.startswith('Test set'):
            pers.append(float(line[-8:-3]))
    return pers


pers = read_directory(sys.argv[1])
plt.legend()
plt.show()

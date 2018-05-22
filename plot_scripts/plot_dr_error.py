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
            # print(pers)
            plt.plot(pers, label=f)
    return pers


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    pers = []
    for line in content:
        if line.startswith('pp'):
            a = line.split()
            pers.append(float(a[2]))
        if line.startswith('np'):
            a = line.split()
            pers[-1] += float(a[2])
    return pers[2:]


pers = read_directory(sys.argv[1])
plt.legend()
plt.show()

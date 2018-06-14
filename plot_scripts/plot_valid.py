import sys
from os import listdir
from os.path import isfile, join
# import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt


def read_directory(dir_name):

    titles = ['test square error', 'test square error std',
              'test normalized square error',
              'test normalized square error std',
              'test accuracy', 'test balanced accuracy', 'test auc score',
              'test precision', 'test recall', 'test f1 score',
              'training loss', 'validation loss', 'validation ls loss',
              'validation logistic loss', 'validation sigmoid loss']

    to_plot = [[] for _ in range(15)]
    plot_or_not = [False for _ in range(15)]
    names = []

    for f in listdir(dir_name):
        if isfile(join(dir_name, f)):
            curves = read_one_file(join(dir_name, f))
            print(f)
            names.append(f)
            for i in range(15):
                if curves[i] != []:
                    plot_or_not[i] = True
                    to_plot[i].append(curves[i])

    for i in [1, 2, 3, 5, 7, 8, 13, 14]:
        plot_or_not[i] = False

    for i in range(15):
        if plot_or_not[i]:
            plt.figure()
            plt.title(titles[i])
            for j, curve in enumerate(to_plot[i]):
                # curve = scipy.ndimage.filters.gaussian_filter1d(curve, 2)
                plt.plot(curve, label=names[j])
            plt.legend()


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    errs, err_stds, n_errs, n_err_stds = [], [], [], []
    accs, b_accs, aucs = [], [], []
    pres, recls, f1s = [], [], []
    losses, val_losses = [], []
    val_ls_losses, val_log_losses, val_sig_losses = [], [], []
    for i, line in enumerate(content):
        # if line == '\n':
            # errs, err_stds, n_errs, n_err_stds = [], [], [], []
            # accs, b_accs, aucs = [], [], []
            # pres, recls, f1s = [], [], []
            # losses, val_losses = [], []
            # val_ls_losses, val_log_losses, val_sig_losses = [], [], []
        if line.startswith('Test set: Error:'):
            a = line.split()
            errs.append(float(a[3]))
        if line.startswith('Test set: Error Std:'):
            a = line.split()
            err_stds.append(float(a[4]))
        if line.startswith('Test set: Normalized Error:'):
            a = line.split()
            n_errs.append(float(a[4]))
        if line.startswith('Test set: Normalized Error Std:'):
            a = line.split()
            n_err_stds.append(float(a[5]))
        if line.startswith('Test set: Accuracy:'):
            a = line.split()
            try:
                accs.append(float(a[3][:-1]))
            except:
                try:
                    accs.append(float(line[-8:-3]))
                except:
                    accs.append(float(line[-7:-3]))
        if line.startswith('Test set: Balanced Accuracy:'):
            a = line.split()
            b_accs.append(float(a[4][:-1]))
        if line.startswith('Test set: Auc Score:'):
            a = line.split()
            aucs.append(float(a[4][:-1]))
        if line.startswith('Test set: Precision:'):
            a = line.split()
            pres.append(float(a[3][:-1]))
        if line.startswith('Test set: Recall Score:'):
            a = line.split()
            recls.append(float(a[4][:-1]))
        if line.startswith('Test set: F1 Score:'):
            a = line.split()
            f1s.append(float(a[4][:-1]))
        if line.startswith('Epoch'):
            a = line.split()
            losses.append(float(a[4]))
        if (line.startswith('Validation Loss')
                and content[i+1].startswith('Epoch')):
            a = line.split()
            val_losses.append(float(a[2]))
        if (line.startswith('Validation Ls Loss')
                and (content[i+3].startswith('Epoch'))):
            a = line.split()
            val_ls_losses.append(float(a[3]))
        if (line.startswith('Validation Log Loss')
                and (content[i+2].startswith('Epoch'))):
            a = line.split()
            val_log_losses.append(float(a[3]))
        if (line.startswith('Validation Sig Loss')
                and (content[i+1].startswith('Epoch'))):
            a = line.split()
            val_sig_losses.append(float(a[3]))
    return (errs, err_stds, n_errs[2:], n_err_stds[2:],
            accs, b_accs, aucs, pres, recls, f1s,
            losses, val_losses,
            val_ls_losses, val_log_losses, val_sig_losses)


read_directory(sys.argv[1])
plt.show()

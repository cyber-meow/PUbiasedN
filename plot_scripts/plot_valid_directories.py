import argparse
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('directory_path')
parser.add_argument('--dataset', action='store', default='mnist')

args = parser.parse_args()
dataset = args.dataset


def read_directory(dir_name):

    plot_all = False
    to_plot = {}

    titles = ['test square error', 'test square error std',
              'test normalized square error',
              'test normalized square error std',
              'test accuracy', 'test balanced accuracy', 'test auc score',
              'test precision', 'test recall', 'test f1 score',
              'training loss', 'validation loss', 'validation ls loss',
              'validation logistic loss', 'validation sigmoid loss']
    plot_or_not = [False for _ in range(15)]

    for d in listdir(dir_name):
        if d != 'ignored' and isdir(join(dir_name, d)):
            to_plot[d] = [[] for _ in range(15)]
            for f in listdir(join(dir_name, d)):
                if isfile(join(dir_name, d, f)):
                    curves = read_one_file(join(dir_name, d, f))
                    print(f)
                    for i in range(15):
                        if curves[i] != []:
                            plot_or_not[i] = True
                            to_plot[d][i].append(curves[i])  # [:202])
    for i in [2, 3, 5, 7, 8, 13, 14]:
        plot_or_not[i] = False

    for i in range(15):
        if plot_or_not[i]:
            plt.figure()
            plt.title(titles[i])
            for lab in to_plot:
                if plot_all:
                    for j, curve in enumerate(to_plot[lab][i]):
                        plt.plot(curve, label='{}_{}'.format(lab, j))
                else:
                    if to_plot[lab][i] != []:
                        m = np.mean(np.array(to_plot[lab][i]), axis=0)
                        # m = scipy.ndimage.filters.gaussian_filter1d(m, 1)
                        s = np.std(np.array(to_plot[lab][i]), axis=0)
                        plt.plot(m, label=lab)
                        plt.fill_between(
                                np.arange(len(m)), m-s/2, m+s/2, alpha=0.5)
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
                and i != len(content)-1
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


read_directory(args.directory_path)
plt.show()

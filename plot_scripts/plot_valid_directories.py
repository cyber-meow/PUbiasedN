import argparse
from os import listdir
from os.path import isfile, isdir, join

import numpy as np
# import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sbn
# from cycler import cycler


sbn.set()
# sbn.set(font_scale=2)

parser = argparse.ArgumentParser()

parser.add_argument('directory_path')
parser.add_argument('--dataset', action='store', default='mnist')
parser.add_argument('--save', action='store')

args = parser.parse_args()
dataset = args.dataset


def read_directory(dir_name):

    pi = 0.56
    plot_all = False
    to_plot = {}

    titles = ['test square error', 'test square error std',
              'test normalized square error',
              'test normalized square error std',
              'Test Error', 'test balanced accuracy', 'test auc score',
              'test precision', 'test recall', 'test f1 score',
              'training loss', 'validation loss', 'validation ls loss',
              'validation logistic loss', 'validation sigmoid loss',
              'false positive rate',
              'False Positive Rate',
              'False Negative Rate', 'negative rate (computed)']
    plot_or_not = [False for _ in range(19)]

    for d in listdir(dir_name):
        if d != 'ignored' and isdir(join(dir_name, d)):
            to_plot[d] = [[] for _ in range(19)]
            for f in listdir(join(dir_name, d)):
                if isfile(join(dir_name, d, f)):
                    curves = read_one_file(join(dir_name, d, f))
                    print(f)
                    for i in range(16):
                        if curves[i] != []:
                            plot_or_not[i] = True
                            to_plot[d][i].append(curves[i])  # [:202])
                    try:
                        acs = np.array(to_plot[d][4][-1])
                        rcls = np.array(to_plot[d][8][-1])
                        fns = 100 - (acs - rcls * pi)/(1-pi)
                        to_plot[d][16].append(fns)
                        plot_or_not[16] = True
                        to_plot[d][17].append(100-rcls)
                        plot_or_not[17] = True
                        ns = pi * (100-rcls) + (acs - rcls * pi)
                        to_plot[d][18].append(ns)
                        plot_or_not[18] = True
                        to_plot[d][4][-1] = 100 - acs
                    except:
                        pass
    for i in [1, 2, 3, 5, 7, 8, 13, 14, 15, 18]:
        plot_or_not[i] = False

    # plt.rc('axes', prop_cycle=(
    #     cycler('color', ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])))

    print('')
    for i in range(19):
        if i == 4 or i == 16 or i == 17 or i == 11:
            print(titles[i])
        if plot_or_not[i]:
            plt.figure()
            plt.ylabel(titles[i])
            plt.xlabel('Epoch')
            plt.tight_layout()
            for lab in sorted(to_plot.keys()):
                if plot_all:
                    for j, curve in enumerate(to_plot[lab][i]):
                        plt.plot(curve, label='{}_{}'.format(lab, j))
                else:
                    if to_plot[lab][i] != []:
                        m = np.mean(np.array(to_plot[lab][i]), axis=0)
                        # m = scipy.ndimage.filters.gaussian_filter1d(m, 1)
                        s = np.std(np.array(to_plot[lab][i]), axis=0)
                        line = plt.plot(m[:-1], label=lab)
                        plt.fill_between(
                            np.arange(len(m)-1),
                            m[:-1]-s[:-1]/2, m[:-1]+s[:-1]/2, alpha=0.5)
                        plt.axhline(m[-1], ls='-.', lw='1.2',
                                    color=line[0].get_color())
                        if i == 4 or i == 16 or i == 17:
                            print(lab)
                            print(m[-1], 100-m[-1], s[-1])
                        if i == 11:
                            print(lab)
                            print(np.min(m))
            # if i == 4:
            #     plt.ylim(ymax=30)
            #     # plt.legend()
            # if i == 16:
            #     plt.ylim(ymax=30)
            # if i == 17:
            #     plt.ylim(ymax=65)
            # plt.xlim(xmin=0, xmax=100)
            plt.legend()
            if i == 4 and args.save is not None:
                plt.ylim(ymin=80)
                plt.savefig(args.save)


def read_one_file(filename):
    with open(filename) as f:
        content = f.readlines()
    errs, err_stds, n_errs, n_err_stds = [], [], [], []
    accs, b_accs, aucs = [], [], []
    pres, recls, f1s = [], [], []
    losses, val_losses = [], []
    val_ls_losses, val_log_losses, val_sig_losses = [], [], []
    fprs = []
    for i, line in enumerate(content):
        if line == '\n':
            errs, err_stds, n_errs, n_err_stds = [], [], [], []
            accs, b_accs, aucs = [], [], []
            pres, recls, f1s = [], [], []
            losses, val_losses = [], []
            val_ls_losses, val_log_losses, val_sig_losses = [], [], []
            fprs = []
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
        if line.startswith('Test set: False Positive Rate:'):
            a = line.split()
            fprs.append(float(a[5][:-1]))
    return (errs, err_stds, n_errs[2:], n_err_stds[2:],
            accs, b_accs, aucs, pres, recls, f1s,
            losses, val_losses,
            val_ls_losses, val_log_losses, val_sig_losses, fprs)


read_directory(args.directory_path)
# plt.show()

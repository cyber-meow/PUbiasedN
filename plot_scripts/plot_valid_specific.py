import argparse
from os import listdir
from os.path import isfile, join
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
    names = ['partial_n', 'ls_partial_n', 'pu_est_partial_n',
             'ls05_partial_n', 'gradual_eta', 'ls_gradual_eta',
             'perfect_partial_n', 'identify_n', 'rho0', 'rho-0',
             'hard_label', 'sampling', 'p_some_u_plus_n', 'pu_plus_n',
             'perfect03_partial_n', 'perfect05_partial_n',
             'nnpu', 'nn-pu', 'log_nnpu', 'pu+n', 'pu+-n', 'pu_then_pn',
             'three_class_prob', 'three_class_max',
             'unbiased_pn', 'pn', 'iwpn',
             'pu_prob_est', 'pu_prob_sig_est', 'sig_partial_n',
             'n2pu_prob_est', 'n2pu_prob_sig_est', 'ls_prob_est',
             'ls_cal', 'n2pu_cal']

    titles = ['test square error', 'test square error std',
              'test normalized square error',
              'test normalized square error std',
              'test accuracy', 'test auc score', 'test f1 score',
              'training loss', 'validation loss', 'validation ls loss',
              'validation logistic loss', 'validation sigmoid loss']
    plot_or_not = [False for _ in range(12)]

    for f in listdir(dir_name):
        if f != 'all' and isfile(join(dir_name, f)):
            curves = read_one_file(join(dir_name, f))
            print(f)
            for name in names:
                if f.startswith('{}_{}'.format(dataset, name)):
                    if name not in to_plot:
                        to_plot[name] = [[] for _ in range(11)]
                    for i in range(12):
                        if (curves[i] != []
                                and not (i == 5 and name == 'pu_then_pn')):
                            plot_or_not[i] = True
                            to_plot[name][i].append(curves[i])
    for i in [2, 3]:
        plot_or_not[i] = False

    for i in range(12):
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
    accs, aucs, f1s = [], [], []
    losses, val_losses = [], []
    val_ls_losses, val_log_losses, val_sig_losses = [], [], []
    for i, line in enumerate(content):
        if line == '\n':
            errs, err_stds, n_errs, n_err_stds = [], [], [], []
            accs, aucs = [], []
            losses, val_losses = [], []
            val_ls_losses, val_log_losses, val_sig_losses = [], [], []
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
            try:
                accs.append(float(line[-8:-3]))
            except:
                accs.append(float(line[-7:-3]))
        if line.startswith('Test set: Auc Score:'):
            aucs.append(float(line[-8:-3]))
        if line.startswith('Test set: F1 Score:'):
            try:
                f1s.append(float(line[-8:-3]))
            except:
                f1s.append(float(line[-7:-3]))
        if line.startswith('valid'):
            a = line.split()
            if len(val_losses) <= 2:
                val_losses.append(float(a[1]))
            else:
                val_losses.append(float(a[1])*1+val_losses[-1]*0)
        if line.startswith('Epoch'):
            a = line.split()
            losses.append(float(a[4]))
        if (line.startswith('Validation Loss')
            and (content[i+1].startswith('Epoch')
                 or (dataset == 'cifar10'
                     and not content[i+1].startswith('Validation')
                     and content[i+10].startswith('Epoch')))):
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
            accs, aucs, f1s, losses, val_losses,
            val_ls_losses, val_log_losses, val_sig_losses)


read_directory(args.directory_path)
plt.show()

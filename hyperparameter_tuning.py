#!/usr/bin/python

import time
import itertools
from subprocess import call


output_file_name = 'census_100P+100N(nnpnu)'

# lrs = [1e-2, 1e-3]
lrs = [1e-3]
# wds = [1e-2, 1e-3, 1e-4]
wds = [1e-2]
# rhos = [0.2, 0.3]
rhos = [0]
# upers = [0.5, 0.7, 0.9]
upers = [0.7]
# gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
gammas = [0.5]

for lr, wd, rho, uper, gamma, i in\
        itertools.product(lrs, wds, rhos, upers, gammas, range(21, 71)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_rho_{}_uper_{}_gamma_{}_{}\n'
        .format(output_file_name, lr, wd, rho, uper, gamma, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_rho_{}_uper_{}_gamma_{}_{}\n'
        .format(output_file_name, lr, wd, rho, uper, gamma, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --rho {} --u_per {} --gamma {} --adjust_p True'
        .format(i, lr, wd, rho, uper, gamma))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)

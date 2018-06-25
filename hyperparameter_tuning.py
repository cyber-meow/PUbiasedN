#!/usr/bin/python

import time
import itertools
from subprocess import call


output_file_name = 'cal_housing_50P(nnpu)'

lrs = [1e-2, 1e-3, 1e-4]
wds = [1e-2, 1e-3, 1e-4]
# rhos = [0.2, 0.3]
rhos = [0]
# upers = [0.5, 0.7, 0.9]
upers = [0.5]
# aps = [True, False]
aps = [False]

for lr, wd, rho, uper, ap, i in\
        itertools.product(lrs, wds, rhos, upers, aps, range(11, 21)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_rho_{}_uper_{}_ap_{}_{}\n'
        .format(output_file_name, lr, wd, rho, uper, ap, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_rho_{}_sep_{}_ap_{}_{}\n'
        .format(output_file_name, lr, wd, rho, uper, ap, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --rho {} --u_per {} --adjust_p {}'
        .format(i, lr, wd, rho, uper, ap))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)

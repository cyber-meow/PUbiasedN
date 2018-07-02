#!/usr/bin/python

import time
from subprocess import call


output_file_name = 'bank_100P+100N'

lr = 1e-3
wd = 1e-2
rho = 0.2
uper = 0.7
gamma = 0.5
algo = 0

for i in range(21, 71):

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
         --rho {} --u_per {} --gamma {} --adjust_p True --algo {}'
        .format(i, lr, wd, rho, uper, gamma, algo))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)

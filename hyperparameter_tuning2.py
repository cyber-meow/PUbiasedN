#!/usr/bin/python

import time
import itertools
from subprocess import call


dataset_name = 'bank'

lrs = [1e-2, 1e-3]
wds = [1e-2, 1e-3, 1e-4]
rho = 0.2
upers = [0.5, 0.7, 0.9]
gammas = [0.1, 0.3, 0.5, 0.7, 0.9]


output_file_name = '{}_100P'.format(dataset_name)

for lr, wd, uper, i in\
        itertools.product(lrs, wds, upers, range(11, 21)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_uper_{}_{}\n'
        .format(output_file_name, lr, wd, uper, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_uper_{}_{}\n'
        .format(output_file_name, lr, wd, uper, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --rho {} --u_per {} --adjust_p True --algo 0'
        .format(i, lr, wd, 0, uper))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)


output_file_name = '{}_100P+100N'.format(dataset_name)

for lr, wd, uper, i in\
        itertools.product(lrs, wds, upers, range(11, 21)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_uper_{}_{}\n'
        .format(output_file_name, lr, wd, uper, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_uper_{}_{}\n'
        .format(output_file_name, lr, wd, uper, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --rho {} --u_per {} --adjust_p True --algo 0'
        .format(i, lr, wd, rho, uper))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)


output_file_name = '{}_100P(nnpu)'.format(dataset_name)

for lr, wd, i in\
        itertools.product(lrs, wds, range(11, 21)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_{}\n'
        .format(output_file_name, lr, wd, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_{}\n'
        .format(output_file_name, lr, wd, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --adjust_p True --algo 1'
        .format(i, lr, wd))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)


output_file_name = '{}_100P+100N(nnpnu)'.format(dataset_name)

for lr, wd, gamma, i in\
        itertools.product(lrs, wds, gammas, range(11, 21)):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write(
        '#$ -N {}_lr_{}_wd_{}_gamma_{}_{}\n'
        .format(output_file_name, lr, wd, gamma, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write(
        '#$ -o running_results/{}_lr_{}_wd_{}_gamma_{}_{}\n'
        .format(output_file_name, lr, wd, gamma, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write(
        '../anaconda3/bin/python pu_biased_n.py\
         --random-seed {} --learning_rate {} --weight_decay {}\
         --gamma {} --adjust_p True --algo 2'
        .format(i, lr, wd, gamma))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(0.1)

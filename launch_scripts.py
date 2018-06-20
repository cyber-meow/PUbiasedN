#!/usr/bin/python

import time
from subprocess import call


output_file_name = 'cifar10_2000P'

for i in range(1, 6):

    job_sub = open('job_sub.sh', 'w')

    job_sub.write('#!/bin/bash\n')
    job_sub.write('#$ -S /bin/bash\n')
    job_sub.write('#$ -N {}_{}\n'.format(output_file_name, i))
    job_sub.write('#$ -cwd\n')
    job_sub.write('#$ -ac d=nvcr-pytorch-1802\n')
    job_sub.write('#$ -jc gpu-container_g1.24h\n')
    job_sub.write('#$ -o running_results/{}_{}\n'
                  .format(output_file_name, i))
    job_sub.write('#$ -j y\n\n')

    job_sub.write('/fefs/opt/dgx/env_set/common_env_set.sh\n')
    job_sub.write('../anaconda3/bin/python pu_biased_n.py --random-seed {}'
                  .format(i))
    job_sub.close()
    call(['qsub', 'job_sub.sh'])
    time.sleep(1)

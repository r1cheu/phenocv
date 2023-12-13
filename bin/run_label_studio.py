#!/usr/bin/env python3
import os
import subprocess

host_ip = os.environ.get('host_ip')
sam_weight = os.environ.get('sam_weight')
label_anything = os.environ.get('label_anything')

cmd = ('nohup label-studio-ml start sam --port 8005 --with sam_config=vit_b '
       f'sam_checkpoint_file={sam_weight} out_bbox=True '
       'out_mask=False device=cuda:7 model_name=sam_hq >/dev/null 2>&1 &')

cmd2 = (f'nohup label-studio start --internal-host {host_ip} --port 8080'
        ' --no-browser >/dev/null 2>&1 &')

if __name__ == '__main__':
    os.chdir(label_anything)
    subprocess.run(cmd, shell=True)
    subprocess.run(cmd2, shell=True)

#!/usr/bin/env python3
import tensorflow as tf
import os
import sys
from tfprocess import TFProcess

with open(sys.argv[1], 'r') as f:
    weights = []
    for e, line in enumerate(f):
        if e == 0:
            #Version
            print("Version", line.strip())
            if line != '1\n':
                raise ValueError("Unknown version {}".format(line.strip()))
        else:
            weights.append(list(map(float, line.split(' '))))
        if e == 2:
            channels = len(line.split(' '))
            print("Channels", channels)
    blocks = e - (4 + 14)
    if blocks % 8 != 0:
        raise ValueError("Inconsistent number of weights in the file")
    blocks /= 8
    print("Blocks", blocks)

x = [
    tf.placeholder(tf.float32, [None, 18, 19 * 19]),
    tf.placeholder(tf.float32, [None, 362]),
    tf.placeholder(tf.float32, [None, 1])
    ]

tfprocess = TFProcess(x, channels, blocks)
tfprocess.replace_weights(weights)
path = os.path.join(os.getcwd(), "leelaz-model")
save_path = tfprocess.saver.save(tfprocess.session, path, global_step=0)

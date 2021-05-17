# coding: utf-8

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app

import sys
import numpy as np
import datetime
import time
import logging
import tensorflow.compat.v1 as tf

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning import training_utils

from depth_and_motion_learning.configs import cfg as gcfg
from depth_and_motion_learning.configs import view_api


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    app_run_beg = time.time()
    training_utils.predict(depth_motion_field_model.predict_input_fn_ex,
                           depth_motion_field_model.infer_depth,
                           depth_motion_field_model.get_vars_to_restore_fn)

    app_run_end = time.time()
    logging.warning('===> elaspsed {} seconds'.format(
        app_run_end - app_run_beg))
    print('done @{}'.format(datetime.datetime.now()))
    logging.warning('done @{}'.format(datetime.datetime.now()))


if __name__ == '__main__':
    print('sys.version:     {}'.format(sys.version))
    print('np.__version__:  {}'.format(np.__version__))
    print('tf.__version__:  {}'.format(tf.__version__))
    print('start @{}'.format(datetime.datetime.now()))
    app.run(main)

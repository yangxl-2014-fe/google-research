# coding: utf-8

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import app

import sys
import numpy as np
import datetime
import tensorflow.compat.v1 as tf

from depth_and_motion_learning import depth_motion_field_model
from depth_and_motion_learning import training_utils

from depth_and_motion_learning.configs import cfg as gcfg
from depth_and_motion_learning.configs import view_api


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  training_utils.predict(depth_motion_field_model.input_fn_predict,
                       depth_motion_field_model.infer_depth,
                       depth_motion_field_model.get_vars_to_restore_fn)


if __name__ == '__main__':
    print('sys.version:     {}'.format(sys.version))
    print('np.__version__:  {}'.format(np.__version__))
    print('tf.__version__:  {}'.format(tf.__version__))
    print('start @{}'.format(datetime.datetime.now()))
    app.run(main)


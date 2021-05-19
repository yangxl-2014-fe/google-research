# coding=utf-8

"""
功能:
----
将 Freetech 摄像头采集的数据转换为 struct2depth 格式的训练数据.

输入:
----
1. 图像目录
2. 内参文件
3. 顺序还是倒序 (采集模式含 前视、后视、侧视)

步骤:
----
1. 生成三合一的小图及内参文件.

参考:
----
yangxl-2014-fe/my_forked/models_struct2depth/research/struct2depth/gen_data_kitti.py

日志:
----
Author:   YangXL
Date:     2021/05/19
Update:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import glob
import argparse

from absl import app
from absl import flags

import cv2
import numpy as np


FLAGS = flags.FLAGS
flags.DEFINE_string('dir_input', '', 'path to freetech captured image sequences.')
flags.DEFINE_string('file_intr_calib', '', 'path to intrinsic calibrated file.')
flags.DEFINE_string('dir_output', '', 'path to struct2depth format data.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('dir_input:       {}'.format(FLAGS.dir_input))
  print('file_intr_calib: {}'.format(FLAGS.file_intr_calib))
  print('dir_output:      {}'.format(FLAGS.dir_output))


def parse_args():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description='Depth and Motion Learning')
    parser.add_argument('--input_dir',
                        help='path to freetech captured image sequences',
                        default='/disk4t0/Fast-Exp/mono-depth-freetech-captured/1',
                        required=False)
    parser.add_argument('--input_calib',
                        help='path to intrinsic calibrated file',
                        default='/disk4t0/Fast-Exp/mono-depth-freetech-captured/CameraParams.yml',
                        required=False)
    parser.add_argument('--output_dir',
                        help='path to saving struct2depth format data',
                        default='/disk4t0/Fast-Exp/mono-depth-freetech-captured/struct2depth-format',
                        required=False)
    parser.add_argument('--img_ext',
                        help='extension format of images',
                        default='.jpeg')
    parser.add_argument('--seq_length',
                        help='image sequence length',
                        default=3,
                        required=False)
    parser.add_argument('--width',
                        help='image width',
                        default=416,
                        required=False)
    parser.add_argument('--height',
                        help='image width',
                        default=128,
                        required=False)
    parser.add_argument('--stepsize',
                        help='step size',
                        default=1,
                        required=False)
    return parser.parse_args()

def get_intr_info(file_calib):
    fs = cv2.FileStorage(file_calib, cv2.FILE_STORAGE_READ)
    intr_info = dict()
    mat = np.zeros((3, 3), dtype=np.float32)
    mat[0][0] = float(fs.getNode('fx').real())
    mat[1][1] = float(fs.getNode('fy').real())
    mat[0][2] = float(fs.getNode('cu').real())
    mat[1][2] = float(fs.getNode('cv').real())
    intr_info['mat'] = mat
    intr_info['h'] = int(fs.getNode('imgH').real())
    intr_info['w'] = int(fs.getNode('imgW').real())
    k = np.zeros((3, 1), dtype=np.float32)
    p = np.zeros((2, 1), dtype=np.float32)
    k[0][0] = float(fs.getNode('k1').real())
    k[1][0] = float(fs.getNode('k2').real())
    k[2][0] = float(fs.getNode('k3').real())
    p[0][0] = float(fs.getNode('p1').real())
    p[1][0] = float(fs.getNode('p2').real())
    intr_info['k'] = k
    intr_info['p'] = p
    return intr_info


def run(params):
    """ 合成三小图格式. """
    intr_info = get_intr_info(params.input_calib)
    files = glob.glob(params.input_dir + '/*' + params.img_ext)
    files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
    files = sorted(files)

    seqname = params.input_dir.split('/')[-1]
    save_dir = osp.join(params.output_dir, seqname)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    ct = 0  # image count
    for i in range(params.seq_length, len(files)+1, params.stepsize):
        img_num = str(ct).zfill(10)
        if osp.exists(osp.join(save_dir, img_num + '.png')):
            ct += 1
            continue

        big_img = np.zeros(shape=(params.height, params.width*params.seq_length, 3))
        big_img_seg = np.zeros(shape=(params.height, params.width*params.seq_length, 3))
        wct = 0  # window count number
        calib_representation = ''
        for j in range(i-params.seq_length, i):
            img = cv2.imread(files[j])
            orig_height, orig_width, _ = img.shape

            zoom_x = params.width / orig_width
            zoom_y = params.height / orig_height

            calib_current = intr_info['mat'].copy()
            calib_current[0, 0] *= zoom_x
            calib_current[0, 2] *= zoom_x
            calib_current[1, 1] *= zoom_y
            calib_current[1, 2] *= zoom_y

            calib_representation = ','.join([str(c) for c in calib_current.flatten()])
            img = cv2.resize(img, (params.width, params.height))
            big_img[:, wct*params.width:(wct+1)*params.width] = img
            wct += 1
        cv2.imwrite(save_dir + '/' + img_num + '.png', big_img)
        cv2.imwrite(save_dir + '/' + img_num + '-fseg.png', big_img_seg)
        with open(save_dir + '/' + img_num + '_cam.txt', 'w') as f:
            f.write(calib_representation)
        ct += 1
        pass
    pass



if __name__ == "__main__":
    # app.run(main)
    args = parse_args()
    run(args)
    pass

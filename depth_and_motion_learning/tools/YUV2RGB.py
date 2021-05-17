import argparse
import os
import sys
import glob
import shutil
import cv2
import numpy as np
tag = 'Freetech'
version = '0.1'
data = '2018.07.02'

def walk_dirs(path):
	print('loading dirs...')
	sub_dirs = []
	path_len = len(path)
	for root, dirs, files in os.walk(path):
		sub_dir = root[path_len:]
		if len(sub_dir) == 0:
			sub_dir = '.'
		elif sub_dir[0]=='\\':
			sub_dir = sub_dir[1:]
		if len(sub_dir) == 0:
			sub_dir = '.'
		sub_dirs.append(sub_dir)
		print('dir: \'%s\''%(sub_dir))
	return sub_dirs

def count_files(path, ext):
	print('counting files...')
	count = 0
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(ext):
				sys.stdout.write('%d\r'%(count))
				count += 1
	return count

def walk_images(path, ext):
	images = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(ext):
				images.append(file)
	return images
	
def doing(src_dir, dst_dir, quality):
	#load sub dir
	sub_dirs = walk_dirs(src_dir) 
	img_num  = count_files(src_dir, '.yuv')
	count = 0

	for sub_dir in sub_dirs:
		filter = os.path.join(src_dir, sub_dir, '*.yuv')
		img_names = glob.glob(filter)
		img_names = [os.path.splitext(os.path.basename(img_name))[0] for img_name in img_names]
		img_names.sort()
		if not os.path.exists(os.path.join(dst_dir, sub_dir)):
			os.makedirs(os.path.join(dst_dir, sub_dir))
		
		for img_name in img_names:
			sys.stdout.write('doing... [%d/%d]\r'%(count, img_num))
			sys.stdout.flush()
			count +=1
			src_pathname = os.path.join(src_dir, sub_dir, img_name+'.yuv')
			dst_pathname = os.path.join(dst_dir, sub_dir, img_name+'.jpeg')
			print('\r')
			if 0:
				size = 1280*1920
				yuv = np.fromfile(src_pathname, dtype=np.uint8)
				y = yuv[0:size].reshape(1280,1920)
				u = yuv[size::2]
				u = np.repeat(u, 2, 0).reshape(640, 1920)
				u = np.repeat(u, 2 ,0)
				v = yuv[size+1::2]
				v = np.repeat(v, 2, 0).reshape(640, 1920)
				v = np.repeat(v, 2 ,0)
				print(y.shape, u.shape, v.shape)
				yuv = np.dstack([y,u,v])
				print(yuv.shape)
				rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR, 3)
			else:
				f = open(src_pathname, 'rb')
				yuv = f.read()
				yuv = np.fromstring(yuv, 'B')
				yuv = np.reshape(yuv,(1280 * 3 // 2, 1920))
				rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
			#im = cv2.imread(src_pathname)
			cv2.imwrite(dst_pathname, rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95] ) #[int(cv2.IMWRITE_PNG_COMPRESSION), quality]

		
def parse_args():
	'''parse args'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--src_dir', default=r'\\10.179.2.5\NewPerception\DataCollection\AR0233\side\Right\yuv')
	#parser.add_argument('--src_dir', default=r'D:\01Project\09.DUC\ISP\raw')
	parser.add_argument('--dst_dir', default=r'\\10.179.2.5\NewPerception\DataCollection\AR0233\side\Right\rgb')
	parser.add_argument('--quality',  default=95)
	
	return parser.parse_args()
		
g_sub_dirs = []
if __name__ == '__main__':
	args     = parse_args()
	src_dir = args.src_dir
	dst_dir = args.dst_dir
	quality = args.quality
	
	if 0:
		img = cv2.imread(r'D:\02Develop\temp\20200101_0196.jpeg')
		yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		print(yuv.shape, yuv.dtype)
		exit()
	
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)

	doing(src_dir, dst_dir, quality)
	
	os.system('pause')
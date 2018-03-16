from __future__ import print_function

import sys

import scipy.io as io
import numpy as np
import torch


def torch_to_mat(th_file_path, mat_file_path):
	data = io.savemat(mat_file_path, dict(data=torch.load(th_file_path).numpy()))

if __name__ == '__main__':
	if len(sys.argv) is not 3:
		print('usage: python torch_to_mat.py /path/to/torch/file /path/to/mat/file')
		sys.exit(1)

	th_file_path = sys.argv[1]
	mat_file_path = sys.argv[2]

	torch_to_mat(th_file_path, mat_file_path)
	
	

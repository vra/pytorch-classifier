from __future__ import print_function, division
import sys
import argparse
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models

def param_parser():
	parser = argparse.ArgumentParser(
		description="Test Phase of Spatial ConvNet of Two-Stream by Pytorch")
	parser.add_argument('-model_path', type=str, default='../models/best_model.pt')
	parser.add_argument('-img_path', type=str, required=True)

	return parser

def my_transform():
	test_transform = transforms.Compose([
		transforms.Scale(256),
		transforms.RandomSizedCrop(224),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
		])

	return test_transform


def img_loader(img_name):
	img = Image.open(img_name)
	img = my_transform()(img).float()
	img = Variable(img, requires_grad=True)
	img = img.unsqueeze(0)
	return img.cuda()

def test(model_path, img_path):
	num_labels = 7

	model = torch.load(model_path).cuda()
	img = img_loader(img_path)
	output = model(img)
	m = nn.Softmax()
	output = m(output)
	_, predicted = torch.max(output.data, 1)
	print('predicted:', predicted)
	
def main():
	parser = param_parser()
	args = parser.parse_args()
	
	test(args.model_path, args.img_path)


if __name__ == '__main__':
	main()

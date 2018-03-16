from __future__ import print_function, division

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets, models

from models import FinetuneModel

def args_parse():
	parser = argparse.ArgumentParser(description="Train a Classifier by Pytorch")
	parser.add_argument('-dataset', type=str, required=True)
	parser.add_argument('-tr_pth', type=str, default='../data/imgs/train')
	parser.add_argument('-val_pth', type=str, default='../data/imgs/val')
	parser.add_argument('-batch_size', type=int, default=1)
	parser.add_argument('-conv_arch', type=str, default='resnet152')
	parser.add_argument('-use_gpu', type=bool, default=True)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-momentum', type=float, default=0.7)
	parser.add_argument('-weight_decay', type=float, default=0.0005)
	parser.add_argument('-step_size', type=int, default=10)
	parser.add_argument('-num_epochs', type=int, default=20)
	parser.add_argument('-temp_save_thresh', type=float, default=0.8)

	return parser
	

def data_process(tr_pth, val_pth, batch_size):
	data_transforms = {
		x: transforms.Compose([
		transforms.Scale(256),
		transforms.RandomSizedCrop(224),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5],
							std= [0.5, 0.5, 0.5])
		])
		for x in ['train', 'val']
	}

	data_path = {'train': tr_pth, 'val': val_pth}
	my_datasets = {x: datasets.ImageFolder(root=data_path[x],
		transform=data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(my_datasets[x],
		batch_size=batch_size, shuffle=True, num_workers=1)
		for x in ['train', 'val']}

	dataset_sizes = {x: len(my_datasets[x]) for x in ['train', 'val']}

	return my_datasets, dataloaders, dataset_sizes
	


def finetune_model(model, criterion, optimizer, scheduler, dataloaders,
    dataset_sizes,num_epochs, use_gpu, temp_save_thresh):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                #print('labels.data:', labels.data)
                #print('preds:', preds)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'val' and best_acc > temp_save_thresh:
                temp_model_save_path = 'temp_model_'+str(best_acc)+'.pt'
                torch.save(model, temp_model_save_path)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def main():
    parsers = args_parse()
    args = parsers.parse_args()

    print(args)
    if args.dataset == 'ucf101':
        num_classes = 101
    elif args.dataset == 'hmdb51':
        num_classes = 51

    datasets, dataloaders, dataset_sizes = data_process(args.tr_pth,
        args.val_pth, args.batch_size)

    model = getattr(torchvision.models, args.conv_arch)(pretrained=True)
    if args.conv_arch.startswith('resnet'):
    #    num_ftrs = model.fc.in_features
    #    model.fc = nn.Linear(num_ftrs, num_classes)
        model = FinetuneModel(model, num_classes)
    elif args.conv_arch.startswith('vgg16'):
        #num_ftrs = model.features
        num_ftrs = 25088
        model.ft_dp0 = nn.Dropout()
        model.ft_fc1 = nn.Linear(num_ftrs, 4096)
        model.ft_relu1 = nn.ReLU(inplace=True)
        model.ft_dp1 = nn.Dropout()
        model.ft_fc2 = nn.Linear(4096, 4096)
        model.ft_relu2 = nn.ReLU(inplace=True)
        model.ft_fc3 = nn.Linear(4096, num_classes)

    if args.use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optim_ft = optim.SGD(model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optim_ft,
        step_size=args.step_size, gamma=0.1)
    best_model, best_accu = finetune_model(model, criterion, optim_ft, exp_lr_scheduler,
        dataloaders, dataset_sizes, args.num_epochs, args.use_gpu, args.temp_save_thresh)
    best_model_save_path = 'best_model_{}_{:4f}.pt'.format(args.dataset, best_accu)
    torch.save(best_model, best_model_save_path)


if __name__ == '__main__':
	main()
    

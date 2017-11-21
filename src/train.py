from __future__ import print_function
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn

# Custom generator for our dataset
from src.dataset.datasetLoader import PileupDataset, TextColor
from src.models.wideResNet import Model

'''Train the model and return'''


def train(train_file, depth, widen_factor, drop_rate, batch_size, epoch_limit, learning_rate, l2, debug_mode, gpu_mode,
          seq_len, iteration_jump, num_classes):
    # Convert the image to tensor
    transformations = transforms.Compose([transforms.ToTensor()])

    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    # Load training data
    train_data_set = PileupDataset(train_file, transformations)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=gpu_mode
                              )
    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    # Create the model to train
    input_channels = 4
    model = Model(input_channels, depth, num_classes, widen_factor, drop_rate)
    if gpu_mode:
        model = torch.nn.DataParallel(model).cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2)

    # Train the Model
    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)

    total_loss = 0
    total_images = 0

    for epoch in range(epoch_limit):

        for i, (images, labels) in enumerate(train_loader):
            # If batch size not distributable among all GPUs then skip
            if gpu_mode is True and images.size(0) % 8 != 0:
                continue

            # Take images and labels of batch_size to train
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)
            if gpu_mode:
                images = images.cuda()
                labels = labels.cuda()

            # Go through the window size of the image
            for row in range(0, images.size(2), iteration_jump):
                # Segmentation of image to window size. Currently using seq_len
                if row + seq_len > images.size(2):
                    continue

                x = images[:, :, row:row + seq_len, :]
                y = labels[:, row:row + seq_len]

                # Subsampling step, if there's less evidence of training on het, hom-alt then don't train
                total_variation = torch.sum(y.eq(2)).data[0]
                total_variation += torch.sum(y.eq(3)).data[0]

                if total_variation == 0 and np.random.uniform(0, 1) * 100 > 5:
                    continue
                elif np.random.uniform(0, 1) < total_variation / batch_size < 0.02:
                    continue

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(x)

                loss = criterion(outputs.contiguous().view(-1, 4), y.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                # Add up loss and total trianing set
                total_loss += loss.data[0]
                total_images += (x.size(0) * seq_len)

            if debug_mode:
                sys.stderr.write(TextColor.CYAN + "EPOCH: " + str(epoch) + " Batches done: " + str(i + 1))
                sys.stderr.write(" Loss: " + str(total_loss / total_images) + "\n" + TextColor.END)
                print(str(epoch) + "\t" + str(i + 1) + "\t" + str(total_loss/total_images))

        # After each epoch print loss and save model if debug mode is on
        if debug_mode:
            sys.stderr.write(TextColor.CYAN + 'EPOCH: ' + str(epoch))
            sys.stderr.write(' Loss: ' + str(total_loss / total_images) + "\n" + TextColor.END)
            print(str(epoch) + "\t" + str(total_loss / total_images))

    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)

    return model
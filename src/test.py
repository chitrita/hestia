from __future__ import print_function
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn

from src.dataset.datasetLoader import PileupDataset, TextColor


def most_common(lst):
    return max(set(lst), key=lst.count)


def test(model, test_file, batch_size, num_classes, gpu_mode, seq_len, debug_mode):
    # Convert the image to tensor
    transformations = transforms.Compose([transforms.ToTensor()])

    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)

    # Load the dataset
    test_dataset = PileupDataset(test_file, transformations)
    testloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16,
                            pin_memory=gpu_mode  # CUDA only
                            )

    if debug_mode:
        sys.stderr.write(TextColor.PURPLE + 'Data loading finished\n' + TextColor.END)

    if gpu_mode:
        model = model.cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Change model to 'eval' mode (BN uses moving mean/var).
    model.eval()

    # Create a confusion tensor for evaluation
    confusion_tensor = torch.zeros(num_classes, num_classes)
    test_loss = 0
    total_datapoint = 0
    # Start testing the model
    for counter, (images, labels) in enumerate(testloader):

        # Load images and labels
        images = Variable(images, volatile=True)
        pl = labels
        if gpu_mode:
            images = images.cuda()
        window = 1

        # We take vote between all predicted classes for a base, prediction_stack keeps track of the vote
        prediction_stack = []

        # Per sequence window
        for row in range(0, images.size(2), 1):

            # For now, don't thinking about trailing bases, they are always almost 0s
            if row + seq_len > images.size(2):
                continue

            x = images[:, :, row:row + seq_len, :]
            ypl = pl[:, row]

            # Get prediction in probability
            preds = model(x)
            labels = Variable(pl[:, row:row + seq_len], volatile=True)
            if gpu_mode:
                labels = labels.cuda()

            print(type(preds), type(labels))
            # Calculate the loss and keep track of it
            loss = criterion(preds.contiguous().view(-1, num_classes), labels.contiguous().view(-1))
            test_loss += loss.data[0]
            total_datapoint += (seq_len * images.size(0))

            # Convert probability to class
            preds = preds.data.topk(1)[1]

            # Append the generated prediction to stack so we can vote
            prediction_stack.append(preds)

            # If we are at a position which is higher than window size then we can vote and predict
            if row + 1 >= seq_len:
                # Go through each object in the stack
                for i in range(images.size(0)):
                    pr = []
                    k = seq_len - 1
                    # Collect all the votes
                    for j in range(len(prediction_stack)):
                        pr.append(prediction_stack[j][i][k][0])
                        k -= 1
                    # The most frequent class wins and is the predicted class
                    p = most_common(pr)
                    # Get the target class
                    t = ypl[i]
                    # Update confusion tensor between target and predicted class
                    confusion_tensor[t][p] += 1

                # After we are done pop the top most vote cause we don't need that anymore
                prediction_stack.pop(0)
                # if debug_mode:
                # print(confusion_tensor)

    correctly_predicted = torch.sum(confusion_tensor.diag())
    total_datapoints = torch.sum(confusion_tensor)
    accuracy = 100 * correctly_predicted / total_datapoints
    total_test_loss = test_loss / total_datapoint

    if debug_mode:
        sys.stderr.write(TextColor.RED + 'ACCURACY: ' + str(accuracy) + "\n" + TextColor.END)
        sys.stderr.write(TextColor.RED + 'CONFUSION TENSOR: '+ str(confusion_tensor) + "\n" + TextColor.END)
        print('Accuracy: ', accuracy)
        print('Confusion meter:\n', confusion_tensor)

    return {'loss': total_test_loss, 'accuracy': accuracy}



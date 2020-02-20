import statistics

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
# Scipy is required by GoogleNet
import scipy
from params import test_params
from pathlib import Path

# GOOGLE NET AND INCEPTION NOTES:
# OUR CROSS ENTROPY LOSS DOES NOT USE THE ADDITIONAL INFORMATION FROM GOOGLE NET AUX STUFF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results_dir = './results/'
min_image_size = 299


def init():
    train = True
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for i, param in enumerate(test_params):
        if param['skip']:
            print("Skipping test: " + param['name']['folder'] + ' [%d / %d]' % ((i + 1), len(test_params)))
            continue
        print("Starting test: " + param['name']['folder'] + ' [%d / %d]' % ((i + 1), len(test_params)))
        (train_loader, test_loader) = load_data(param)

        if train:
            (train_times, epoch_performances) = train_network(train_loader, param)
            with open(get_result_path(param) + 'train_times.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['time'])
                for train_time in train_times:
                    writer.writerow([str(train_time)])

            with open(get_result_path(param) + 'train_performance_over_epoch.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['loss', 'accuracy'])
                for (loss, accuracy) in epoch_performances:
                    writer.writerow([str(loss), str(accuracy)])

        (forward_prop_times, accuracy, accuracy_per_class) = test_network(test_loader, classes, param)

        with open(get_result_path(param) + 'network_info.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name', 'epochs', 'batch_size', 'criterion', 'optimizer'])
            writer.writerow(
                [param['name']['network'], param['epochs'], param['batch_size'], param['name']['criterion'],
                 param['name']['optimizer']])

        with open(get_result_path(param) + 'forward_propagation_times.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time'])
            for forward_prop_time in forward_prop_times:
                writer.writerow([str(forward_prop_time)])

        with open(get_result_path(param) + 'accuracy.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['accuracy', *classes])
            writer.writerow([accuracy, *accuracy_per_class])


def get_result_path(params):
    directory = results_dir + params['name']['folder'] + '/'
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_network_path(params):
    return get_result_path(params) + 'network.pth'


def show_image(img):
    img = img / 2 + 0.5  # un normalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


def load_data(params):
    transform = transforms.Compose(
        [transforms.Resize(min_image_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = params['batch_size']

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4)

    return train_loader, test_loader


def fix_logits_output(output, params):
    # Removes auxiliary data from GoogleNet and Inception outputs :(
    if params['name']['network'] in ['GoogleNet', 'InceptionV3']:
        output = output.logits
    return output


def train_network(data_loader, params):
    print("Using device: " + device.type)

    net = params['network']()
    net.to(device)

    criterion = params['criterion']()
    optimizer = params['optimizer'](net.parameters())

    print("Network training started")
    times = []
    epoch_performances = []
    losses = []

    for epoch in range(params['epochs']):  # loop over the data set multiple times
        start_time = time.perf_counter()

        running_loss = 0.0
        correct = 0
        total = 0
        for batch_index, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = fix_logits_output(outputs, params)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loss_value = loss.item()
            losses.append(loss_value)
            running_loss += loss_value
            if batch_index % 100 == 99:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f, acc %.3f %%' %
                      (epoch + 1, batch_index + 1, running_loss / 100, 100 * correct / total))
                running_loss = 0.0

        epoch_performances.append((statistics.mean(losses), correct / total))
        losses = []
        times.append(time.perf_counter() - start_time)

    # Save network
    network_path = get_network_path(params)
    torch.save(net.state_dict(), network_path)
    print('Finished Training, saving network to ' + network_path)
    return times, epoch_performances


def test_network(data_loader, classes, params):
    print("Testing network: " + params['name']['folder'])
    network_path = get_network_path(params)

    net = params['network']()
    net.to(device)
    # Load network
    net.load_state_dict(torch.load(network_path))

    correct = 0
    total = 0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    batch_count = len(data_loader)

    times = []
    with torch.no_grad():
        for batch_index, data in enumerate(data_loader):
            # Check accuracy
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Measure performance of forward propagation
            start_time = time.perf_counter()
            outputs = net(images)
            outputs = fix_logits_output(outputs, params)
            times.append(time.perf_counter() - start_time)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Check accuracy per label
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            if batch_index % 250 == 249:
                print('Test [%d / %d]' % (batch_index + 1, batch_count))

    accuracy = correct / total
    print('Network accuracy: %d %%' % (100 * accuracy))

    accuracy_per_class = []
    for i in range(10):
        class_accuracy = class_correct[i] / class_total[i]
        accuracy_per_class.append(class_accuracy)
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_accuracy))

    return times, accuracy, accuracy_per_class


if __name__ == '__main__':
    init()

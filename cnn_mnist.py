import os
import argparse
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_mnist_data(batch_size):
    print('--> Loading MNIST data...')
    transform = transforms.Compose([
            transforms.Resize((48,48)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST
        ])
    # Get Train and Validation Loaders
    train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx), num_workers=4)

    # Test Loader
    test_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, valid_loader, test_loader

def compute_accuracy(loader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

def train(net, eval_net, train_loader, valid_loader, epochs, save, device):
    optimizer = optim.Adam(net.parameters())
    critirion = nn.CrossEntropyLoss()
    max_valid_acc = 0.0

    print('--> Start training...')
    start = timeit.default_timer()
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = critirion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('--> Epoch %d Step %5d loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        eval_net.load_state_dict(net.state_dict())
        valid_acc = compute_accuracy(valid_loader, eval_net, device)
        print('--> Accuracy of validation: %.4f' % valid_acc)
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            torch.save(net.state_dict(), save)
    stop = timeit.default_timer()
    print('### Running time: %.4f', stop - start)  
    print('### Best accuracy on validation set: %.4f' % max_valid_acc)

    return  max_valid_acc, stop - start
        
def test(net, test_loader, save, device):
    net.load_state_dict(torch.load(save))
    test_acc = compute_accuracy(test_loader, net, device)
    print('### Accuracy of testing data: %.4f' % test_acc)
    return test_acc

def exp(args, device):
    valid_accs = []
    test_accs = []
    running_times = []
    for exp_id in range(args.times):
        train_loader, valid_loader, test_loader = prepare_mnist_data(batch_size=args.batch_size)
        # Train the network
        net = Net().to(device)
        eval_net = Net().to(device)
        max_valid_acc, running_time = train(net, eval_net, train_loader, valid_loader, args.epochs, str(exp_id) + '_' + args.save, device)

        # Record results
        valid_accs.append(max_valid_acc)
        running_times.append(running_time)
        total_params = sum(p.numel() for p in net.parameters())

        # Load the trained model and test it on testing data
        net = Net().to(device)
        test_acc = test(net, test_loader, str(exp_id) + '_' + args.save, device)
        test_accs.append(test_acc)

    print('!!! Total number of parameters', total_params)
    print('!!! Training time mean %.4f, std %.4f' % (np.mean(running_times), np.std(running_times)))
    print('!!! Validation accuracy mean %.4f, std %.4f' % (np.mean(valid_accs), np.std(valid_accs)))
    print('!!! Testing accuracy mean %.4f, std %.4f' % (np.mean(test_accs), np.std(test_accs)))

def parse_args():
    parser = argparse.ArgumentParser(description='CNN on MNIST')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--times', default=5, type=int)
    parser.add_argument('--save', default='cnn_mnist.pth', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    # Train and evaluate
    exp(args, device)

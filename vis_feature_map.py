import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.model import Net
from demo import train
import tqdm

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset import RandomRotate
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='ORN.PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use-arf', action='store_true', default=True,
                    help='upgrading to ORN')
parser.add_argument('--orientation', type=int, default=8, metavar='O',
                    help='nOrientation for ARFs (default: 8)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

kwargs = {'num_works': 1, 'pin_memory': True} if args.cuda else {}
mnist_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
  transforms.Scale(32), RandomRotate((-180, 180)), transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))]))

mnist_test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([
  transforms.Scale(32), RandomRotate((-180, 180)), transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(
    mnist_train_dataset,
    num_workers=2,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    mnist_test_dataset,
    num_workers=2,
    batch_size=8,
    shuffle=True
)

model = Net(args.use_arf, args.orientation)
criterion = F.nll_loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 200
for epoch in range(1, epochs+1):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


# Plot some images
idx = torch.randint(0, output.size(0), ())
pred = normalize_output(output[idx, 0])
img = data[idx, 0]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(img.detach().numpy())
axarr[1].imshow(pred.detach().numpy())

# Visualize feature maps
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.conv1.register_forward_hook(get_activation('conv1'))
data, _ = mnist_test_dataset[0]
data.unsqueeze_(0)
output = model(data)

act = activation['conv1'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].imshow(act[idx])
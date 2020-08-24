import torch.nn as nn
import torch.nn.functional as F
from functions import rotation_invariant_encoding
from modules import ORConv2d


class Net(nn.Module):

  def __init__(self, use_arf=False, nOrientation=8):
    super(Net, self).__init__()
    self.use_arf = use_arf
    self.nOrientation = nOrientation
    if use_arf:
      self.conv1 = ORConv2d(1, 10, arf_config=(1,nOrientation), kernel_size=3)
      self.conv2 = ORConv2d(10, 20, arf_config=nOrientation,kernel_size=3)
      self.conv3 = ORConv2d(20, 40, arf_config=nOrientation,kernel_size=3, stride=1, padding=1)
      self.conv4 = ORConv2d(40, 80, arf_config=nOrientation,kernel_size=3)
    else:
      self.conv1 = nn.Conv2d(1, 80, kernel_size=3)
      self.conv2 = nn.Conv2d(80, 160, kernel_size=3)
      self.conv3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
      self.conv4 = nn.Conv2d(320, 640, kernel_size=3)
    self.fc1 = nn.Linear(640, 1024)
    self.fc2 = nn.Linear(1024, 10)


  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    x = F.relu(self.conv4(x))
    if self.use_arf:
        x = rotation_invariant_encoding(x, self.nOrientation)
    x = x.view(-1, 640)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)
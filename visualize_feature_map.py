import torch
import torch.nn as nn
from model import Net
import matplotlib.pyplot as plt

model = Net()


def visualize_feature_map(path):
    model = Net()
    model.load_state_dict(torch.load(path))

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save each of the conv layers in this list

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    plt.figure(figsize=(30, 30))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(26, 26, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('../visualize_layers/filter.png')
    plt.show()

if __name__ == "__main__":
    visualize_feature_map()
import torch
import torch.nn as nn
from model import Net
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from dataset import RandomRotate


def visualize_feature_map(path, img, img_num):
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

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving image {img_num} layer {num_layer} feature maps...")
        plt.savefig(f"../outputs/{img_num}layer_{num_layer}.png")
        # plt.show()
        plt.close()


if __name__ == "__main__":

    mnist_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.Scale(32), RandomRotate((-180, 180)), transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(mnist_train_dataset,
                                               batch_size=1,
                                               shuffle=True)

    img_num = 0
    for img, _ in train_loader:
        visualize_feature_map('model/model_state.pth', img, img_num)
        img += 1
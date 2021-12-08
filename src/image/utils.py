import matplotlib.pyplot as plt
import numpy as np
import torch


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def weights_init_xavier(m):
    # Copied from https://github.com/JohnEfan/PyTorch-ALAD/blob/6e7c4a9e9f327b5b08936376f59af2399d10dc9f/utils/utils.py#L4
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.zero_()

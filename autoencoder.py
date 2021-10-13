"""
https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/scripts/3.1.simple%20autoencoder%202D.py
"""

import os
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from nibabel import load as load_fmri
from scipy.stats import scoreatpercentile

import torch
from torch import nn, no_grad
from torch.nn import functional
from torch import optim
from torch.autograd import Variable

from torch.utils.data import DataLoader
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd

from data_declaration import Task
from loader_helper import LoaderHelper


# class CustomizedDataset(Dataset):
#     def __init__(self, data_root):
#         self.samples = []
#
#         for item in glob(os.path.join(data_root, '*.nii.gz')):
#             self.samples.append(item)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         temp = load_fmri(self.samples[idx]).get_data()
#         temp[temp < scoreatpercentile(temp.flatten(), 2)] = 0
#         max_weight = temp.max()
#         temp = temp / max_weight
#         return temp, max_weight


class Encoder2D(nn.Module):
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding_mode='zeros',
                 pool_kernal_size=2,
                 ):
        super(Encoder2D, self).__init__()

        # self.batch_size = batch_size
        if out_channels is None:
            out_channels = [128, 256, 512]
        if in_channels is None:
            in_channels = [166, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.pool_kernal_size = pool_kernal_size

        self.conv2d_66_128 = nn.Conv2d(in_channels=self.in_channels[0],
                                       out_channels=self.out_channels[0],
                                       kernel_size=self.kernel_size,
                                       stride=self.stride,
                                       padding_mode=self.padding_mode,
                                       )
        self.conv2d_128_128 = nn.Conv2d(in_channels=self.in_channels[1],
                                        out_channels=self.out_channels[0],
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding_mode=self.padding_mode,
                                        )
        self.conv2d_128_256 = nn.Conv2d(in_channels=self.in_channels[1],
                                        out_channels=self.out_channels[1],
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding_mode=self.padding_mode,
                                        )
        self.conv2d_256_256 = nn.Conv2d(in_channels=self.in_channels[2],
                                        out_channels=self.out_channels[1],
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding_mode=self.padding_mode,
                                        )
        self.conv2d_256_512 = nn.Conv2d(in_channels=self.in_channels[2],
                                        out_channels=self.out_channels[2],
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding_mode=self.padding_mode,
                                        )
        self.conv2d_512_512 = nn.Conv2d(in_channels=self.in_channels[3],
                                        out_channels=self.out_channels[2],
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding_mode=self.padding_mode,
                                        )

        self.activation = nn.CELU(inplace=True)
        self.output_activation = nn.Sigmoid()
        self.pooling = nn.AvgPool2d(kernel_size=self.pool_kernal_size,
                                    stride=2,
                                    )
        self.output_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.norm128 = nn.BatchNorm2d(num_features=128)
        self.norm256 = nn.BatchNorm2d(num_features=256)
        self.norm512 = nn.BatchNorm2d(num_features=512)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out1 = self.norm128(self.conv2d_66_128(x))
        out1 = self.activation(out1)
        out1 = self.dropout(out1)
        out1 = self.norm128(self.conv2d_128_128(out1))
        out1 = self.activation(out1)
        out1 = self.pooling(out1)

        out2 = self.norm256(self.conv2d_128_256(out1))
        out2 = self.activation(out2)
        out2 = self.dropout(out2)
        out2 = self.norm256(self.conv2d_256_256(out2))
        out2 = self.activation(out2)
        out2 = self.pooling(out2)

        out3 = self.norm512(self.conv2d_256_512(out2))
        out3 = self.activation(out3)
        out3 = self.dropout(out3)
        out3 = self.norm512(self.conv2d_512_512(out3))
        out3 = self.output_activation(out3)
        out3 = self.pooling(out3)

        flatten = torch.squeeze(self.output_pooling(out3))

        return flatten


class Decoder2D(nn.Module):
    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=7,
                 stride=1,
                 padding_mode='zeros', ):
        super(Decoder2D, self).__init__()

        # self.batch_size = batch_size
        if out_channels is None:
            out_channels = [512, 256, 128, 166]
        if in_channels is None:
            in_channels = [512, 256, 128, 166]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode

        self.convT2d_512_512 = nn.ConvTranspose2d(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )
        self.convT2d_512_256 = nn.ConvTranspose2d(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )

        self.convT2d_256_256 = nn.ConvTranspose2d(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )
        self.convT2d_256_128 = nn.ConvTranspose2d(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels[2],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )

        self.convT2d_128_128 = nn.ConvTranspose2d(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels[2],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )
        self.convT2d_128_66 = nn.ConvTranspose2d(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels[3],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding_mode=self.padding_mode, )

        self.convT2d_66_66 = nn.ConvTranspose2d(in_channels=self.in_channels[3],
                                                out_channels=self.out_channels[
                                                    3],
                                                kernel_size=self.kernel_size,
                                                stride=self.stride,
                                                padding_mode=self.padding_mode,)

        self.activation = nn.CELU(inplace=True)
        self.output_activation = nn.Sigmoid()
        self.norm512 = nn.BatchNorm2d(num_features=512)
        self.norm256 = nn.BatchNorm2d(num_features=256)
        self.norm128 = nn.BatchNorm2d(num_features=128)
        self.norm66 = nn.BatchNorm2d(num_features=66)

    def forward(self, x):
        reshaped = x.view(self.batch_size, 512, 1, 1)

        out1 = self.norm512(self.convT2d_512_512(reshaped))
        out1 = self.norm512(self.convT2d_512_512(out1))
        out1 = self.norm256(self.convT2d_512_256(out1))
        out1 = self.activation(out1)
        out1 = functional.interpolate(out1, size=(24, 24))

        out2 = self.norm256(self.convT2d_256_256(out1))
        out2 = self.norm256(self.convT2d_256_256(out2))
        out2 = self.norm128(self.convT2d_256_128(out2))
        out2 = self.activation(out2)
        out2 = functional.interpolate(out2, size=(48, 48))

        out3 = self.norm128(self.convT2d_128_128(out2))
        out3 = self.norm128(self.convT2d_128_128(out3))
        out3 = self.norm66(self.convT2d_128_66(out3))
        out3 = self.activation(out3)
        # no need to interpolate because it is 66 x 66 x 66

        out4 = self.norm66(self.convT2d_66_66(out3))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.activation(out4)
        out4 = functional.interpolate(out4, size=(88, 88))

        return out4


def createLossAndOptimizer(net, learning_rate=1e-3):
    # Loss function
    loss = nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                           weight_decay=1e-9)

    return loss, optimizer


def train_loop(net, lr, dataloader, device, n_epochs):
    """
    A for-loop of train the autoencoder for 1 epoch
    """
    loss_func, optimizer = createLossAndOptimizer(net, learning_rate=lr)
    pbar = tqdm(dataloader)
    for epoch in range(n_epochs):
        train_loss = 0.
        net.train()
        for ii, batch in enumerate(pbar):
            # if ii + 1 < len(dataloader):
            # load the data to memory
            batch = tuple(item.to(device) for item in batch)
            batch_x, batch_y = batch
            # one of the most important step, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs = net(batch_x)
            # compute the losses
            loss_batch = loss_func(outputs, batch_x)
            loss_batch += 0.01 * torch.norm(outputs, 1)
            selected_params = torch.cat([x.view(-1) for x in net.parameters()])
            loss_batch += 0.01 * torch.norm(selected_params, 1)
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss += loss_batch.item()
            pbar.set_description(
                f'Epoch {epoch}/{n_epochs},loss = {train_loss / (ii + 1):.3f}')
    return net


def validation_loop(model, dataloader, device):
    # specify the gradient being frozen
    loss_func = nn.SmoothL1Loss()
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        for ii, batch in tqdm(enumerate(dataloader)):
            # load the data to memory
            batch = tuple(item.to(device) for item in batch)
            batch_x, batch_y = batch
            # compute the outputs
            outputs = model(batch_x)
            # compute the losses
            loss_batch = loss_func(outputs, batch_x)
            # record the validation loss of a mini-batch
            valid_loss += loss_batch.data
            denominator = ii
        valid_loss = valid_loss / (denominator + 1)
    print(f'validation loss = {valid_loss:.3f}')
    return valid_loss


def main():
    saving_name = '../results/simple_autoencoder2D.pth'

    batch_size = 2
    lr = 1e-5  # was 1e-5
    n_epochs = 20
    seed = 444
    print(f'set up random seeds: {seed}')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print('set up data loaders')
    ld_helper = LoaderHelper(task=Task.CN_v_AD)
    train_dl = ld_helper.get_train_dl(batch_size=batch_size)
    test_dl = ld_helper.get_test_dl(batch_size=1)
    print('set up model')
    encoder = Encoder2D()
    decoder = Decoder2D()
    model = nn.Sequential(OrderedDict(
        [('encoder', encoder),
         ('decoder', decoder),
         ]
    )).to(device)
    # if os.path.exists(saving_name.replace(".pth", ".csv")):
    #     model.load_state_dict(torch.load(saving_name))
    #     model.eval()
    #     results = pd.read_csv(saving_name.replace(".pth", ".csv"))
    #     results = {col_name: list(results[col_name].values) for col_name in
    #                results.columns}
    #     best_valid_loss = torch.tensor(results['valid_loss'][-1],
    #                                    dtype=torch.float64)
    #     stp = 1 + len(results['valid_loss'])
    # else:
    print('Initialize')
    results = dict(
        train_loss=[],
        valid_loss=[],
        epochs=[])
    best_valid_loss = torch.from_numpy(np.array(np.inf))
    # stp = 1

    model = train_loop(model, lr, train_dl, device, n_epochs)
    # validation
    print('validating ...')
    valid_loss = validation_loop(model, test_dl, device)
    torch.save(model.state_dict(), saving_name)
    print('saving model')


if __name__ == '__main__':
    main()

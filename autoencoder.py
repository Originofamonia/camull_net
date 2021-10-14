"""
https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/scripts/3.1.simple%20autoencoder%202D.py
"""

import os
import math
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
import pickle

import torch
from torch import nn, no_grad
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from torch.optim import lr_scheduler
from collections import OrderedDict
import torch
import numpy as np
import pandas as pd

from data_declaration import Task
from loader_helper import LoaderHelper

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


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
                                    stride=2,)
        self.output_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.norm128 = nn.BatchNorm2d(num_features=128)
        self.norm256 = nn.BatchNorm2d(num_features=256)
        self.norm512 = nn.BatchNorm2d(num_features=512)
        self.dropout = nn.Dropout2d(p=0.4)

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
                                                padding_mode=self.padding_mode, )

        self.activation = nn.CELU(inplace=True)
        self.output_activation = nn.Sigmoid()
        self.norm512 = nn.BatchNorm2d(num_features=512)
        self.norm256 = nn.BatchNorm2d(num_features=256)
        self.norm128 = nn.BatchNorm2d(num_features=128)
        self.norm66 = nn.BatchNorm2d(num_features=self.out_channels[-1])

    def forward(self, x):
        reshaped = x.view(-1, self.in_channels[0], 1, 1)

        out1 = self.norm512(self.convT2d_512_512(reshaped))
        out1 = self.norm512(self.convT2d_512_512(out1))
        out1 = self.norm256(self.convT2d_512_256(out1))
        out1 = self.activation(out1)
        out1 = F.interpolate(out1, size=(24, 24))

        out2 = self.norm256(self.convT2d_256_256(out1))
        out2 = self.norm256(self.convT2d_256_256(out2))
        out2 = self.norm128(self.convT2d_256_128(out2))
        out2 = self.activation(out2)
        out2 = F.interpolate(out2, size=(48, 48))

        out3 = self.norm128(self.convT2d_128_128(out2))
        out3 = self.norm128(self.convT2d_128_128(out3))
        out3 = self.norm66(self.convT2d_128_66(out3))
        out3 = self.activation(out3)
        # no need to interpolate because it is 66 x 66 x 66

        out4 = self.norm66(self.convT2d_66_66(out3))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.activation(out4)
        out4 = F.interpolate(out4, size=(256, 256))

        return out4


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder2D()
        self.decoder = Decoder2D()

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding

        x = self.encoder(x).view(-1, 2, -1)

        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        # x = F.relu(self.dec1(z))
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var


def createLossAndOptimizer(args, net):
    # Loss function
    loss = nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    return loss, optimizer


def train_ae(args, net, dataloader):
    """
    train the autoencoder
    """
    loss_func, optimizer = createLossAndOptimizer(args, net)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.n_epochs)) / 2) * (
                1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    pbar = tqdm(dataloader)
    for epoch in range(args.n_epochs):
        train_loss = 0.
        net.train()
        for ii, batch in enumerate(pbar):
            # if ii + 1 < len(dataloader):
            # load the data to memory
            batch = tuple(item.to(args.device) for item in batch)
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
            pbar.set_description(f'Epoch {epoch}/{args.n_epochs},loss = {train_loss / (ii + 1):.3f}')
        print(
            f'Epoch {epoch}/{args.n_epochs},loss = {train_loss / (ii + 1):.3f}')
        scheduler.step()
    return net


def evaluate_ae(args, model, test_dl):
    """
    infer features from AE and then classify by logistic regression
    """
    loss_func = nn.SmoothL1Loss()
    model.eval()
    feats = []
    labels = []
    valid_loss = 0.
    with torch.no_grad():
        for ii, batch in tqdm(enumerate(test_dl)):
            # load the data to memory
            batch = tuple(item.to(args.device) for item in batch)
            batch_x, batch_y = batch
            # compute the outputs
            feat = model.encoder(batch_x)
            # print(feat)
            feats.append(feat.squeeze().detach().cpu().numpy())
            labels.append(batch_y.squeeze().detach().cpu().numpy())
            outputs = model.decoder(feat)
            # compute the losses
            loss_batch = loss_func(outputs, batch_x)
            # record the validation loss of a mini-batch
            valid_loss += loss_batch.data
            denominator = ii
        valid_loss = valid_loss / (denominator + 1)
    print(f'validation loss = {valid_loss:.3f}')
    feats = np.array(feats)
    labels = np.array(labels)
    clf = LogisticRegression(random_state=0).fit(feats, labels)
    preds = clf.predict(feats)
    print(f"preds: {preds}")
    print(f"labels: {labels}")
    acc = clf.score(feats, labels)
    print(f"LR acc: {acc}")
    return valid_loss


def main():
    saving_name = 'results/autoencoder2D.pt'
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=444)
    parser.add_argument('--n_epochs', type=int, default=100)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(args.seed)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {args.device}')
    print('set up data loaders')
    ld_helper = LoaderHelper(task=Task.CN_v_AD)
    train_dl = ld_helper.get_train_dl(batch_size=args.batch_size)
    test_dl = ld_helper.get_test_dl(batch_size=1)
    print('set up model')
    encoder = Encoder2D()
    decoder = Decoder2D()
    model = nn.Sequential(OrderedDict(
        [('encoder', encoder),
         ('decoder', decoder),
         ]
    )).to(args.device)
    model.double()

    model = train_ae(args, model, train_dl)
    # validation
    print('Validating:')
    evaluate_ae(args, model, test_dl)
    torch.save(model.state_dict(), saving_name)


if __name__ == '__main__':
    main()

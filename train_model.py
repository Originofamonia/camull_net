"""The following module trains the weights of the neural network model."""
import os
import datetime
import uuid
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from data_declaration import Task
from loader_helper import LoaderHelper
from architecture import load_cam_model, Camull
from evaluation import evaluate_model

# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")
#     print("Running on the GPU.")
# else:
#     DEVICE = torch.device("cpu")
#     print("Running on the CPU")


def save_weights(model_in, uuid_arg, fold=1, task: Task = None):
    """The following function saves the weights file into required folder"""
    # root_path = ""

    if task == Task.CN_v_AD:
        root_path = "weights/NC_v_AD/" + uuid_arg + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + uuid_arg + "/"

    if fold == 1:
        os.mkdir(root_path)  # otherwise it already exists

    while True:
        s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format(
            fold, date=datetime.datetime.now())  # pylint: disable=line-too-long

        if os.path.exists(s_path):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break


def load_model(arch, path=None):
    """Function for loaded camull net from a specified weights path"""
    # if arch == "camull":  # must be camull
    if path is None:
        model = load_cam_model(
            "weights/camnet/fold_0_weights-2020-04-09_18_29_02")
    else:
        model = load_cam_model(path)

    return model


def build_arch(device):
    """Function for instantiating the pytorch neural network object"""
    net = Camull()

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)

    net.to(device)
    net.double()

    return net


def train_loop(model, train_dl, epochs, device):
    """Function containing the neural net model training loop"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    loss_function = nn.BCELoss()
    model.train()

    for i in range(epochs):
        pbar = tqdm(train_dl)
        for _, batch in enumerate(pbar):
            batch = tuple(item.to(device) for item in batch)
            batch_x, batch_y = batch
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'E: {i}/{epochs}, loss: {loss.item():.3f}')

    return model


def train_camull(ld_helper, device, epochs, model=None):
    """The function for training the camull network"""
    task = ld_helper.get_task()
    uuid_ = uuid.uuid4().hex
    # model_cop = model

    # for k_ind in range(k_folds):

    if model is None:
        model = build_arch(device)

    train_dl = ld_helper.get_train_dl(batch_size=2)
    test_dl = ld_helper.get_test_dl(batch_size=1)
    model = train_loop(model, train_dl, epochs, device)
    # save_weights(model, uuid_, fold=k_ind + 1, task=task)
    evaluate(device, model, test_dl)

    print("Completed train_camull.")

    return model


def evaluate(device, model, test_dl):
    model.eval()
    pbar = tqdm(test_dl)
    outputs = []
    targets = []
    for _, batch in enumerate(pbar):
        batch = tuple(item.to(device) for item in batch)
        batch_x, batch_y = batch
        output = model(batch_x)
        outputs.append(output.item())
        targets.append(batch_y.item())
        pbar.set_description(f'Evaluate')

    print(f'targets: {targets}')
    print(f'outputs: {outputs}')
    pickle.dump([outputs, targets], open('evaluate.pkl', 'wb'))


def start(ld_helper, device, epochs=40, model_uuid=None):
    task = ld_helper.get_task()
    if task == Task.CN_v_AD:
        model_uuid = train_camull(ld_helper, device, epochs=epochs)
    else:  # sMCI vs pMCI
        if model_uuid:
            model = load_model("camull", model_uuid)
            model_uuid = train_camull(ld_helper, device, model=model, epochs=epochs)
        else:
            print("Need a model uuid")

    return model_uuid


def main():
    """Main function of the module."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # CN v AD
    ld_helper = LoaderHelper(task=Task.CN_v_AD)
    model = train_camull(ld_helper, device, epochs=4)

    # # transfer learning for pMCI v sMCI
    # ld_helper.change_task(Task.sMCI_v_pMCI)
    # model = load_model("camull", model_uuid)
    # uuid = train_camull(ld_helper, device, model=model, epochs=40)
    # evaluate_model(device, uuid, ld_helper)


if __name__ == '__main__':
    main()

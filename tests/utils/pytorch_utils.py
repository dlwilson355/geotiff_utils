"""
This file provides various utilities related to pytorch including constructing loss data loaders, loss functions,
optimizers, and model training and testing.
"""


import os.path

from alive_progress import alive_bar
import numpy as np
import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from utils.general_utils import list_mean
from utils.geotiff_utils import write_to_geotiff_within_lat_lon, make_arc_gis_ready, create_geotiff


def initialize_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(0)


def print_model(model, input_dimensions):
    summary(model, input_size=(1, *input_dimensions))
    print(model)


def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_optimizer(model):
    return torch.optim.Adam(model.parameters())


def one_hot_encode(tensor, num_classes=2):

    class_dim = len(tensor.size()) - 3
    swap_dim = len(tensor.size()) - 1

    transposed_tensor = torch.transpose(tensor, class_dim, swap_dim)
    squeezed_tensor = torch.squeeze(transposed_tensor, dim=swap_dim)
    one_hot_tensor = torch.nn.functional.one_hot(squeezed_tensor.to(torch.int64), num_classes=num_classes)
    reverse_transposed_one_hot_tensor = torch.transpose(one_hot_tensor, class_dim, swap_dim)
    floating_point_tensor = reverse_transposed_one_hot_tensor.double()

    return floating_point_tensor


def convert_to_labels(tensor):

    labeled_tensor = torch.argmax(tensor, dim=1)

    return labeled_tensor


def run_model(model,
              data_loader,
              loss_fn,
              optimizer=None,
              train=False,
              verbose=False,
              name='running',
              fast_mode=False):
    """
    Enumerates the data loader, sends the data to the model, and calculates the loss.
    If train is true, the gradients will be calculated and the weights of the model updated.
    The optimizer parameter is only required if train is set to True.
    """

    def update_progressbar_text(progressbar, losses, step, accuracy):
        if len(losses) == 0:
            mean_loss = None
            last_loss = None
        else:
            mean_loss = round(list_mean(losses), 2)
            last_loss = round(losses[-1], 2)
        progressbar.text = f'Average Loss: {mean_loss} | '\
                           f'Most Recent Loss: {last_loss} | '\
                           f'Proportion Correct: {accuracy} | Current Step: {step}'

    num_samples = len(data_loader)
    losses = list()
    num_correct = 0
    num_labels = 1
    model.train(train)

    with alive_bar(len(data_loader), dual_line=True, title=f'Running model {name}:') as progressbar:
        update_progressbar_text(progressbar, losses, f'Starting model {name}', None)

        for i, (input, output) in enumerate(data_loader):

            # create a temporary function for quickly updating the progressbar
            fast_progressbar_update = lambda text: \
                    update_progressbar_text(progressbar, losses, text, num_correct/num_labels)

            # forward propagate through the model to generate a prediction
            fast_progressbar_update('Performing forward propagation')
            preds = model.forward(input)['out']
            # fast_progressbar_update_func('Clamping predictions and output tensors between 0 or 1')
            # fast_progressbar_update_func('One hot encoding output')
            if verbose:
                print(torch.unique(output))
            if verbose:
                print('Debug info below...')
                print(output.size())
                print(preds.size())
                print(torch.min(preds))
                print(torch.max(preds))
                print(torch.min(output))
                print(torch.max(output))
                print(output)
                print(preds)
                print("Debug info above...\n\n")
            assert preds.size() == output.size(), \
                f'The predictions from the model and the output dataset should be the same shape.\n' \
                f'Instead the predictions were shape "{preds.size()}" and the output dataset was size ' \
                f'"{output.size()}".'

            # update the loss and backpropagate
            fast_progressbar_update('Calculating loss')
            loss = loss_fn(preds, output)
            if train:
                fast_progressbar_update('Zeroing gradient')
                optimizer.zero_grad()
                fast_progressbar_update('Backpropagating loss')
                loss.backward()
                fast_progressbar_update('Clipping gradients')
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
                fast_progressbar_update('Updating model weights')
                optimizer.step()
            last_loss = loss.item()
            losses.append(last_loss)

            # count the proportion of correct class predictions
            output_labels = convert_to_labels(output)
            preds_labels = convert_to_labels(preds)
            num_correct += torch.sum(preds_labels == output_labels).item()
            num_labels += output_labels.numel()

            progressbar()
            fast_progressbar_update(f'Loading data sample {i+1} of {num_samples}')

            # if fast mode is enabled, we skip the majority of training, this is used for debugging
            if fast_mode and i > 2:
                break

    print(f'Completed model {name}.')

    return losses, num_correct/num_labels


def build_output_geotiff(model,
                         data_loader,
                         save_geotiff_path,
                         convert_to_class_label=False,
                         overwrite=True,
                         name='generating geotiff'):
    """
    Enumerates the data loader, sends the data to the model, retrieves the output, and outputs it into a geotiff file.
    """

    # remove the file if it exists, or if not supposed to overwrite, return from this function
    if os.path.exists(save_geotiff_path):
        if overwrite:
            os.remove(save_geotiff_path)
            print(f'Removed pre-existing output geotiff at: {save_geotiff_path}')
    else:
        return

    # create an empty geotiff to write the data to
    if convert_to_class_label:
        num_bands = 1
    else:
        num_bands = data_loader.dataset.get_num_classes()
    no_data_values = [-128 for _ in range(num_bands)]
    create_geotiff(save_geotiff_path,
                   data_loader.dataset.output_data_path,
                   num_bands=num_bands,
                   no_data_values=no_data_values)

    num_samples = len(data_loader)
    model.train(False)

    # run inference with the model for each data sample and save the outputs to the geotiff file
    with alive_bar(len(data_loader), dual_line=True, title=f'Running model {name}:') as progressbar:
        progressbar.text = f'Starting model {name}.'
        for i, (inputs, _) in enumerate(data_loader):
            progressbar.text = f'Loaded data sample {i} of {num_samples}'
            preds = model.forward(inputs)['out']
            progressbar.text = 'Completed forward propagation'
            if convert_to_class_label:
                preds = convert_to_labels(preds)
            else:
                preds = torch.squeeze(preds, dim=0)
            coords = data_loader.dataset.cell_list[i]
            write_to_geotiff_within_lat_lon(save_geotiff_path, preds, coords[0], coords[1])

            progressbar.text = 'Wrote sample output to geotiff.'
            progressbar()

    print(f'Completed model {name}.')

    make_arc_gis_ready(save_geotiff_path)

    print(f'Made geotiff at "{save_geotiff_path}" ArcGIS ready.')

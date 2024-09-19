"""
This file contains utilities for saving and loading checkpoints including the full models used to build the function,
its weights, the functions used to create the losses, the optimizers, ect.
"""


import os.path

import dill
import torch


class Checkpoint:
    """This class saves a checkpoint with all the associated information."""

    def __init__(self, base_checkpoint_path):
        base, ext = os.path.splitext(base_checkpoint_path)
        if ext == '.pth':
            base_checkpoint_path = base
        self.base_checkpoint_path = base_checkpoint_path

    def save_checkpoint(self,
                        model,
                        model_build_func,
                        optimizer,
                        optimizer_build_func,
                        loss_func,
                        loss=None,
                        epoch=None,
                        notes=None,
                        **kwargs):

        save_dict = {'model_state_dict': model.state_dict(),
                     'model_build_func': dill.dumps(model_build_func),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'optimizer_build_func': dill.dumps(optimizer_build_func),
                     'loss_func': dill.dumps(loss_func),
                     'loss': loss,
                     'epoch': epoch,
                     'notes': notes,
                     'extras': kwargs}

        torch.save(save_dict, self.get_checkpoint_path(epoch))

    def load_checkpoint(self, checkpoint_path="latest"):
        """Loads all relevant information from the model so that it can be recovered for training or inference."""

        # Load the checkpoint file using torch.load()
        if checkpoint_path == "latest":
            checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path is None:
            print(f"No checkpoint to load for checkpointer at: {self.base_checkpoint_path}.")
            return
        checkpoint = torch.load(checkpoint_path)

        # Initialize the model, optimizer, and loss function using the corresponding build functions
        self.model_build_func = dill.loads(checkpoint['model_build_func'])
        self.model = self.model_build_func()
        self.optimizer_build_func = dill.loads(checkpoint['optimizer_build_func'])
        self.optimizer = self.optimizer_build_func(self.model)
        self.loss_func = dill.loads(checkpoint['loss_func'])

        # Load the model and optimizer state dicts from the checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load other information from the checkpoint such as loss, epoch, and notes
        self.loss = checkpoint['loss']
        self.epoch = checkpoint['epoch']
        self.notes = checkpoint['notes']
        self.checkpoint_dict = checkpoint

        # load any additional extras that were saved as part of the checkpoint
        for key in checkpoint['extras'].keys():
            setattr(self, key, checkpoint['extras'][key])

    def add_to_checkpoint(self, **kwargs):
        """
        Adds and arguments passed to this function to the checkpoint, saves them, and loads them into this checkpoint
        object.
        """

        self.save_checkpoint(self.model,
                             self.model_build_func,
                             self.optimizer,
                             self.optimizer_build_func,
                             self.loss_func,
                             self.loss,
                             self.epoch,
                             self.notes,
                             **self.checkpoint_dict['extras'],
                             **kwargs)

        self.load_checkpoint(self.get_checkpoint_path(self.epoch))

    def _delete_checkpoint(self, checkpoint_path):
        if os.path.exists(self.base_checkpoint_path):
            os.remove(self.base_checkpoint_path)
            print(f"Removed checkpoint at: {checkpoint_path}")

    def delete_checkpoints(self, epochs="all"):
        """
        Deletes all checkpoints for the pass integer epoch or list of epochs.
        If epochs is "all" deletes all checkpoints.
        """

        if epochs == "all":
            for epoch in range(100):
                checkpoint_path = self.get_checkpoint_path(epoch)
                self._delete_checkpoint(checkpoint_path)
            return

        if isinstance(epochs, int):
            self._delete_checkpoint(self.get_checkpoint_path(epochs))
            return

        if isinstance(epochs, list):
            for epoch in epochs:
                self._delete_checkpoint(self.get_checkpoint_path(epoch))
            return

        raise ValueError(f'Expected value for "epochs" to be None, int, or list.\n'
                         f'Instead got type {type(epochs)}.')

    def get_checkpoint_path(self, epoch=None):
        """Returns the checkpoint path for the passed epoch."""

        if epoch:
            return self.base_checkpoint_path + f'epoch_{epoch}.pth'
        else:
            return self.base_checkpoint_path + '.pth'

    def get_latest_checkpoint(self):
        """Returns the full path to the saved checkpoint with the greatest epoch."""

        for i in reversed(range(100)):
            checkpoint_path = self.get_checkpoint_path(i)
            if os.path.exists(checkpoint_path):
                return checkpoint_path

        return None

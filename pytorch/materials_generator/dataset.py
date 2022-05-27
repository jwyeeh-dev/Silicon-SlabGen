import torch
import functools
import numpy as np
import pandas as pd
from os import path
from torch.utils.data import Dataset


class MaterialsGeneratorDataset(Dataset):
    """
    Wrapper for a dataset
    """

    def __init__(self, item_ids, data_dir, raw_data_dir):
        """
        Args:
            item_ids (List): materials ids for cell images
            data_dir (string): path for preprocessed data
            raw_data_dir (string): path for raw csv data
        """
        self.item_ids = item_ids
        self.data_dir = data_dir
        self.surface_energy_data = pd.read_csv(raw_data_dir)
        self.cell_image_df = pd.read_csv(path.join(data_dir, 'cell_image.csv'))
        self.basis_image_df = pd.read_csv(path.join(data_dir, 'basis_image.csv'))

    def __len__(self):
        return len(self.item_ids)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        item_id = self.item_ids[idx]
        cell_image_name = self.cell_image_df[self.cell_image_df['item_id'] == item_id]['image_name'].values
        basis_image_name = self.basis_image_df[self.basis_image_df['item_id'] == item_id]['image_name'].values

        # load encoded cell image
        cell_vector = np.zeros((200, 1))
        try:
            cell_image_encoded = np.load(path.join(self.data_dir, 'cell_image_encode', '{}.npy'.format(cell_image_name[0])))
            #print(cell_image_encoded)
            cell_image_encoded = np.expand_dims(cell_image_encoded, axis=1)
            #print(cell_image_encoded.shape)
            cell_vector[0:len(cell_image_encoded), :] = cell_image_encoded
        except IndexError:
            pass
        

        # load encoded cell image
        basis_vector = np.zeros((200, 4))
        for i, name in enumerate(basis_image_name):
            basis_image_encoded = np.load(path.join(self.data_dir, 'basis_image_encode', '{}.npy'.format(name)))
            #print(basis_image_encoded.shape)
            basis_vector[:,i] = basis_image_encoded
        #print(cell_vector.shape)
        #print(basis_vector.shape)
        vector = np.concatenate([basis_vector, cell_vector], axis=1)
        # add a new axis
        vector = vector.reshape((1, 5, 200))
        # reshape (channel, height, width) = (6, 1, 200)
        vector = np.transpose(vector, (1, 0, 2))

        # for surface energy task
        surface_energy = self.surface_energy_data[
            self.surface_energy_data['item_id'] == item_id
        ]['surface_energy'].values[0]
        label = 0 if surface_energy <= -0.5 else 1
        return torch.tensor(vector, dtype=torch.float), torch.tensor([label], dtype=torch.float)

import os
from matplotlib.pyplot import imread
from typing import TYPE_CHECKING, Tuple, Union
import torchvision.transforms as T
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from mlassistant.core.data import ContentLoader
if TYPE_CHECKING:
    from ..config import TBSegmentConfig


class ScanLoader(ContentLoader):
    """ The ScanLoader class """

    def __init__(self, conf: 'TBSegmentConfig', prefix_name: str, data_specification: str):
        super().__init__(conf, prefix_name, data_specification)
        self.is_ready = False
        self._x, self._y = self._load_data(data_specification)
        self.is_ready = True

    def _load_data(self, data_specification: str) -> Tuple[np.ndarray, np.ndarray]:
      montgomery_dir = '/content/drive/MyDrive/Colab Notebooks/Binary Segmentation/Montgomery'
      shenzhen_dir = '/content/drive/MyDrive/Colab Notebooks/Binary Segmentation/Shenzhen'

      montgomery_imgs, shenzhen_imgs = [], []
      montgomery_labels, shenzhen_labels = [], []

      for filename in os.listdir(os.path.join(montgomery_dir, 'Images')):
          image = imread(os.path.join(montgomery_dir, 'Images/' + filename))
          mask = imread(os.path.join(montgomery_dir, 'Masks/FinalMasks/' + filename))
          montgomery_imgs.append(image)
          montgomery_labels.append(mask)

      for filename in os.listdir(os.path.join(shenzhen_dir, 'Images')):
          image = imread(os.path.join(shenzhen_dir, 'Images/' + filename))
          mask = imread(os.path.join(shenzhen_dir, 'Masks/' + filename))
          shenzhen_imgs.append(image[..., 0])
          shenzhen_labels.append(mask)

      x = np.array(shenzhen_imgs)
      y = np.array(shenzhen_labels)

      x_train, x_test, y_train, y_test = train_test_split(
          x, y, test_size=0.1, random_state=17)
      x_train, x_val, y_train, y_val = train_test_split(
          x_train, y_train, test_size=0.1, random_state=17)

      x_train, y_train = self.add_augmentation(x_train, y_train)

      random_permut = torch.randperm(len(x_train))
      x_train = x_train[random_permut]
      y_train = y_train[random_permut]

      visualization_x = np.array(montgomery_imgs)
      visualization_y = np.array(montgomery_labels)
          
      data = {
          'train': (np.expand_dims(x_train, axis=1), np.expand_dims(y_train, axis=1)),
          'val': (np.expand_dims(x_val, axis=1), np.expand_dims(y_val, axis=1)),
          'test': (np.expand_dims(x_test, axis=1), np.expand_dims(y_test, axis=1)),
          'visual': (np.expand_dims(visualization_x, axis=1), 
                      np.expand_dims(visualization_y, axis=1))
      }

      return data[data_specification]

    def get_samples_names(self):
        ''' sample names must be unique, they can be either scan_names or scan_dirs.
        Decided to put scan_names. No difference'''
        return [str(i) for i in range(len(self._x))]

    def get_samples_labels(self):
        return np.zeros(len(self._y))

    def reorder_samples(self, indices, new_names):
        self._x = self._x[indices]
        self._y = self._y[indices]

    def get_views_indices(self):
        return self.get_samples_names(),\
            np.arange(len(self._x)).reshape((len(self._x), 1))

    def get_samples_batch_effect_groups(self):
        pass

    def add_augmentation(self, x, y):
      x, y = torch.from_numpy(x), torch.from_numpy(y)
      hflip_trans = T.RandomHorizontalFlip(p=1)
      rotation_trans = T.RandomRotation(15)

      x = torch.cat((x, hflip_trans(x), rotation_trans(x)))
      y = torch.cat((y, hflip_trans(y), rotation_trans(y)))
      
      return x, y

    def get_placeholder_name_to_fill_function_dict(self):
        """ Returns a dictionary of the placeholders' names (the ones this content loader supports)
        to the functions used for filling them. The functions must receive as input data_loader,
        which is an object of class data_loader that contains information about the current batch
        (e.g. the indices of the samples, or if the sample has many elements the indices of the chosen
        elements) and return an array per placeholder name according to the receives batch information.
        IMPORTANT: Better to use a fixed prefix in the names of the placeholders to become clear which content loader
        they belong to! Some sort of having a mark :))!"""
        return {
            'x': self._get_x,
            'y': self._get_y,
        }

    def _get_x(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray])\
            -> np.ndarray:
        return self._x[samples_inds]

    def _get_y(self, samples_inds: np.ndarray, samples_elements_inds: Union[None, np.ndarray])\
            -> np.ndarray:
        return self._y[samples_inds]

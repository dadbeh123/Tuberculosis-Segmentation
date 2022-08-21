from typing import List, TYPE_CHECKING, Union, Set, OrderedDict as OrdDict
import warnings
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from mlassistant.core.data import DataLoader
from mlassistant.core import Model, ModelIO
from mlassistant.model_evaluation import NormalEvaluator
from ..data import ScanLoader

if TYPE_CHECKING:
    from mlassistant.core.configs import BaseConfig


class LossEvaluator(NormalEvaluator):

    def __init__(self, model: Model, data_loader: DataLoader, conf: 'BaseConfig'):
        super(LossEvaluator, self).__init__(model, data_loader, conf)

        self.phase = conf.phase

        # for summarized information
        self.avg_loss: float = 0
        self._avg_other_losses: OrdDict[str, float] = OrderedDict()
        self.n_received_samples: int = 0
        self._n_received_samples_other_losses: OrdDict[str, int] = OrderedDict()

    def reset(self):
        self.avg_loss = 0
        self._avg_other_losses = OrderedDict()
        self.n_received_samples = 0
        self._n_received_samples_other_losses = OrderedDict()

    def update_summaries_based_on_model_output(self, model_output: ModelIO) -> None:

        n_batch = self.data_loader.get_current_batch_size()

        new_n = self.n_received_samples + n_batch
        self.avg_loss = self.avg_loss * (float(self.n_received_samples) / new_n) + \
            model_output.get('loss', 0.0) * (float(n_batch) / new_n)

        for kw in model_output.keys():
            if 'loss' in kw and kw != 'loss':

                old_avg = self._avg_other_losses.get(kw, 0.0)
                old_n = self._n_received_samples_other_losses.get(kw, 0)

                new_n = old_n + n_batch

                self._avg_other_losses[kw] = \
                    old_avg * (float(old_n) / new_n) + \
                    model_output[kw].detach().cpu() * (float(n_batch) / new_n)
                self._n_received_samples_other_losses[kw] = new_n

    def update_samples_related_details_based_on_model_output(self,
                                                             model_output: ModelIO) -> None:
        self.update_summaries_based_on_model_output(model_output)

    def save_middle_outputs_of_the_model(self, model_output: ModelIO, save_dir: str) -> None:
        warnings.warn('save_middle_outputs_of_the_model has not been implemented for loss evaluator')

    def get_titles_of_evaluation_metrics(self) -> List[str]:
        return ['Loss'] + list(self._avg_other_losses.keys())

    def get_values_of_evaluation_metrics(self) -> List[str]:
        return ['%.4f' % self.avg_loss] + ['%.4f' % loss for loss in self._avg_other_losses.values()]

    def get_loss(self) -> float:
        return self.avg_loss

    def get_samples_results_summaries(self):
        samples_names = self.data_loader.get_samples_names()
        return samples_names, ['' for _ in range(len(samples_names))]

    def get_samples_results_header(self):
        return []

    def get_samples_elements_results_summaries(self):
        warnings.warn('get_samples_elements_results_summaries has not been implemented for loss evaluator')

    def save_model_outputs(self, save_dir: str,
                           classes_to_use: Union[None, Set[int]] = None) -> None:
        scan_loader = self.data_loader.get_content_loader_of_interest(ScanLoader)
        test_set = scan_loader._load_data('visual')
        self.model.eval()
        for i in range(4):
          imgs, masks, preds = [], [], []
          for j in range(5):
            imgs.append(transforms.ToTensor()(np.squeeze(test_set[0][i * 5 + j])).to('cuda').reshape(-1, 100, 100))
            masks.append(transforms.ToTensor()(np.squeeze(test_set[1][i * 5 + j])).to('cuda').reshape(-1, 100, 100))
            preds.append(self.model(imgs[j].unsqueeze(0), None)['result'])
          for j in range(5):
            temp = (preds[j][0, ...].squeeze() > 0.5).float().detach().cpu().numpy()
            mask = np.squeeze(test_set[1][i * 5 + j])
            scan = test_set[0][i * 5 + j]
            Image.fromarray((scan * \
                              255).astype(np.uint8)).save(os.path.join('/content/drive/MyDrive/Outputs',
                              'scan' + str(5 * i + j) + '.png'))
            Image.fromarray((temp * \
                              255).astype(np.uint8)).save(os.path.join('/content/drive/MyDrive/Outputs',
                              'img' + str(5 * i + j) + '.png'))
            Image.fromarray((mask * \
                              255).astype(np.uint8)).save(os.path.join('/content/drive/MyDrive/Outputs',
                              'mask' + str(5 * i + j) + '.png'))
            

        
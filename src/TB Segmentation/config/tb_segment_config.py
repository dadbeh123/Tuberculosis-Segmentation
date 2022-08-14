from torch import nn
from torchvision import transforms
from mlassistant.config import NormalConfig
from ..evaluator.loss_evaluator import LossEvaluator
from ..data import ScanLoader


class TBSegmentConfig(NormalConfig):

    def __init__(self, try_name: str, try_num: int):

        super().__init__(
            data_separation=None,
            try_name=try_name,
            try_num=try_num,
            evaluator_cls=LossEvaluator,
            content_loaders=[('scans', ScanLoader)],
            inp_size=572
        )

        # replaced configs!
        self.batch_size = 5
        self.training_config.iters_per_epoch = None

        # augmentation
        self.training_config.augmentations_dict = {
            'scans_x': nn.Sequential(
                transforms.RandomHorizontalFlip(p=1)
              )
            # 'scans_y': nn.Sequential(
            #     transforms.RandomHorizontalFlip(p=1)
            #   )
        }

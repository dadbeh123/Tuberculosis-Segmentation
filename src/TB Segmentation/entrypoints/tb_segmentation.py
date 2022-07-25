from mlassistant.entrypoint import BaseEntryPoint
from ..models.tb_segmentation import UNet
from ..config import TBSegmentConfig


class EntryPoint(BaseEntryPoint):
    r'''The name of this class **MUST** be `EntryPoint`'''

    def __init__(self):
        super().__init__(TBSegmentConfig(try_name='TB', try_num=1),
                         UNet())

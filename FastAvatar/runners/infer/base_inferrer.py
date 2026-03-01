import torch
from abc import abstractmethod
from accelerate import Accelerator
from accelerate.logging import get_logger

from FastAvatar.runners.abstract import Runner


logger = get_logger(__name__)


class Inferrer(Runner):

    EXP_TYPE: str = None

    def __init__(self):
        super().__init__()

        torch._dynamo.config.disable = True
        self.accelerator = Accelerator()

        self.model : torch.nn.Module = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def device(self):
        return self.accelerator.device

    @abstractmethod
    def _build_model(self, cfg):
        pass

    @abstractmethod
    def infer(self):
        pass

    def run(self):
        self.infer()
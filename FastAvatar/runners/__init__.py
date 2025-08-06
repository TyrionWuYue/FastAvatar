from VGGTAvatar.utils.registry import Registry

REGISTRY_RUNNERS = Registry() 

from .train import *
from .infer import *
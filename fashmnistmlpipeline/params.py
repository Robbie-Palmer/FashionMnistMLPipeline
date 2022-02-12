from inspect import getmembers
from pathlib import Path

import yaml
from pydantic import BaseModel, validator
from torchvision import models as torchvision_models


class Train(BaseModel):
    seed: int
    num_epochs: int
    valid_pct_split: float
    pretrained: bool
    architecture: str
    metric_target_average_fn: str

    @validator('num_epochs')
    def greater_than_zero(cls, value):
        assert value > 0, 'Must be greater than 0'
        return value

    @validator('valid_pct_split')
    def between_zero_and_one(cls, value):
        assert 0 < value < 1, 'Must be between 0 and 1'
        return value

    @validator('architecture')
    def is_torchvision_model_name(cls, value):
        model_names = [name for name, value in getmembers(torchvision_models) if not name.startswith('_')]
        assert value in model_names
        return value

    @validator('metric_target_average_fn')
    def is_supported_metric_average_function(cls, value):
        metric_average_functions = ['micro', 'macro', 'samples', 'weighted']
        assert value in metric_average_functions
        return value


class Params(BaseModel):
    train: Train


with open(Path(__file__).parent / 'params.yml', 'r') as f:
    params_dict = yaml.load(f, Loader=yaml.FullLoader)
params = Params.parse_obj(params_dict)

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter, parent_label, get_image_files
from fastai.metrics import accuracy, F1Score, Precision, Recall, RocAuc, HammingLoss, Jaccard, MatthewsCorrCoef
from fastai.torch_core import set_seed
from fastai.vision.all import cnn_learner, PILImageBW, ImageBlock
from torch import save as save_model
from torchvision import models as torchvision_models

from fashmnistmlpipeline import data_root
from fashmnistmlpipeline.params import params

if __name__ == '__main__':
    train_params = params.train
    set_seed(train_params.seed)
    model_dir = data_root / 'model'
    model_dir.mkdir(exist_ok=True)

    block = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(train_params.valid_pct_split),
                      get_y=parent_label)
    loaders = block.dataloaders(data_root / 'FashionMNIST/processed')

    architecture = getattr(torchvision_models, train_params.architecture)
    metrics = [accuracy,
               F1Score(average=train_params.metric_target_average_fn),
               Precision(average=train_params.metric_target_average_fn),
               Recall(average=train_params.metric_target_average_fn),
               RocAuc(),
               HammingLoss(),
               Jaccard(average=train_params.metric_target_average_fn),
               MatthewsCorrCoef()]
    learner = cnn_learner(loaders, architecture, pretrained=train_params.pretrained, metrics=metrics)

    learner.fit(train_params.num_epochs)
    learner.export(model_dir / 'learner.pkl')
    save_model(learner.model.state_dict(), model_dir / 'model.pth')

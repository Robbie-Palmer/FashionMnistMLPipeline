from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter, parent_label, get_image_files
from fastai.metrics import accuracy, F1Score, Precision, Recall, RocAuc, HammingLoss, Jaccard, MatthewsCorrCoef
from fastai.torch_core import set_seed
from fastai.vision.all import resnet18, cnn_learner, PILImageBW, ImageBlock
from torch import save as save_model

from fashmnistmlpipeline import data_root

if __name__ == '__main__':
    set_seed(42)
    model_dir = data_root / 'model'
    model_dir.mkdir(exist_ok=True)
    block = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                      get_items=get_image_files,
                      splitter=RandomSplitter(0.2),
                      get_y=parent_label)
    loaders = block.dataloaders(data_root / 'FashionMNIST/processed')
    metrics = [accuracy,
               F1Score(average='weighted'),
               Precision(average='weighted'),
               Recall(average='weighted'),
               RocAuc(),
               HammingLoss(),
               Jaccard(average='weighted'),
               MatthewsCorrCoef()]
    learner = cnn_learner(loaders, resnet18, metrics=metrics)
    learner.fit(1)
    learner.export(model_dir / 'learner.pkl')
    save_model(learner.model.state_dict(), model_dir / 'model.pth')

import json

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter, parent_label, get_image_files
from fastai.interpret import ClassificationInterpretation
from fastai.metrics import accuracy, F1Score, Precision, Recall, RocAuc, HammingLoss, Jaccard, MatthewsCorrCoef
from fastai.torch_core import set_seed
from fastai.vision.all import cnn_learner, PILImageBW, ImageBlock
from pandas import DataFrame
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

    validation_result = learner.validate()
    metric_headers = [metric.name for metric in learner.metrics]
    metric_headers.insert(0, 'Loss')
    metrics_dict = dict(zip(metric_headers, validation_result))
    dataset_metrics = dict(validation=metrics_dict)
    with open(model_dir / 'results.json', 'w') as results_file:
        json.dump(dataset_metrics, results_file)

    interp = ClassificationInterpretation.from_learner(learner)
    validation_tile_predictions = DataFrame({"actual": interp.targs, "predicted": interp.decoded})
    validation_tile_predictions.to_csv(model_dir / 'actual_vs_predicted.csv', index=False)

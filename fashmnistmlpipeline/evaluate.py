import json

from fastai.data.transforms import get_image_files
from fastai.learner import load_learner
from pandas import DataFrame

from fashmnistmlpipeline import data_root, model_dir

if __name__ == '__main__':
    test_data_dir = data_root / 'FashionMNIST/test'
    learner = load_learner(model_dir / 'learner.pkl')
    test_dl = learner.dls.test_dl(get_image_files(data_root / 'FashionMNIST/test'), with_labels=True)
    test_result = learner.validate(dl=test_dl)

    metric_headers = [metric.name for metric in learner.metrics]
    metric_headers.insert(0, 'Loss')
    metrics_dict = dict(zip(metric_headers, test_result))
    dataset_metrics = dict(validation=metrics_dict)
    with open(model_dir / 'test_results.json', 'w') as results_file:
        json.dump(dataset_metrics, results_file)

    probs, preds, targs = learner.get_preds(dl=test_dl, with_input=False, with_decoded=True)
    validation_predictions = DataFrame({"actual": targs, "predicted": preds})
    validation_predictions.to_csv(model_dir / 'test_actual_vs_predicted.csv', index=False)

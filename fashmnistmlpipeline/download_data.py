from fastai.vision.core import PILImageBW
from fire import Fire
from torchvision.datasets import FashionMNIST

from fashmnistmlpipeline import data_root


def download_data(train: bool, sub_dir_name: str):
    dataset_class = FashionMNIST
    dataset_dir = data_root / dataset_class.__name__
    dataset_dir.mkdir(exist_ok=True)
    dataset = dataset_class(train=train, root=data_root, download=True)
    sub_dir = dataset_dir / sub_dir_name
    sub_dir.mkdir(exist_ok=True)
    for idx in range(len(dataset)):
        image = PILImageBW.create(dataset.data[idx])
        target_idx = dataset.targets[idx].item()
        target_name = dataset.classes[target_idx].replace('/', '_')
        target_dir = sub_dir / target_name
        target_dir.mkdir(exist_ok=True)
        image_path = target_dir / f'{idx}.png'
        image.save(image_path)


if __name__ == '__main__':
    Fire(download_data)

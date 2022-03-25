from PIL.Image import fromarray
from numpy import array, uint8
from pandas import read_csv

from fashmnistmlpipeline import data_root


def main():
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    kaggle_dir = data_root / 'kaggle'
    train_image_dir = kaggle_dir / 'train'
    train_image_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(kaggle_dir / 'train.csv')
    for (idx, target_idx, *pixel_data) in df.itertuples(name=None):
        image = fromarray(array(pixel_data, dtype=uint8).reshape((28, 28)))
        target_name = classes[target_idx].replace('/', '_')
        target_dir = train_image_dir / target_name
        target_dir.mkdir(exist_ok=True)
        image_path = target_dir / f'{idx}.png'
        image.save(image_path)


if __name__ == '__main__':
    main()

from PIL.Image import fromarray
from numpy import array, uint8
from pandas import read_csv

from fashmnistmlpipeline import data_root


def main():
    kaggle_dir = data_root / 'kaggle'
    test_image_dir = kaggle_dir / 'test'
    test_image_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(kaggle_dir / 'test.csv', header=None)
    images = df[0].apply(lambda series: fromarray(array([uint8(pixel.split('.')[0])
                                                         for pixel in series.split(' ')]).reshape((28, 28))))
    for idx, image in enumerate(images):
        image.save(test_image_dir / f'{idx}.png')


if __name__ == '__main__':
    main()

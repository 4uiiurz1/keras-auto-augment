import random
from keras.preprocessing.image import ImageDataGenerator

from auto_augment import cutout, apply_policy
from utils import *


class Cifar10ImageDataGenerator:
    def __init__(self, args):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True)

        self.means = np.array([0.4914009 , 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.args = args
        if args.auto_augment:
            self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
                ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
                ['Color', 0.4, 3, 'Brightness', 0.6, 7],
                ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
                ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
                ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
                ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
                ['Brightness', 0.9, 6, 'Color', 0.2, 8],
                ['Solarize', 0.5, 2, 'Invert', 0, 0.3],
                ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
                ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
                ['Color', 0.9, 9, 'Equalize', 0.6, 6],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                ['Brightness', 0.1, 3, 'Color', 0.7, 0],
                ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
                ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
            ]

    def standardize(self, x):
        x = x.astype('float32') / 255

        means = self.means.reshape(1, 1, 1, 3)
        stds = self.stds.reshape(1, 1, 1, 3)

        x -= means
        x /= (stds + 1e-6)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                               seed, save_to_dir, save_prefix, save_format, subset)

        while True:
            x_batch, y_batch = next(batches)

            if self.args.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.args.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):
                    x_batch[i] = apply_policy(x_batch[i], self.policies[random.randrange(len(self.policies))])

            x_batch = self.standardize(x_batch)

            yield x_batch, y_batch


def main():
    import argparse
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10

    parser = argparse.ArgumentParser()
    parser.add_argument('--cutout', default=True, type=str2bool)
    parser.add_argument('--auto-augment', default=True, type=str2bool)
    args = parser.parse_args()

    datagen = Cifar10ImageDataGenerator(args)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    for imgs, _ in datagen.flow(x_train, y_train):
        plt.imshow(imgs[0].astype('uint8'))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()

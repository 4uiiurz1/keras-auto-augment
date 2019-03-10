import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import keras
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from utils import *
from wide_resnet import *
from cosine_annealing import *
from dataset import Cifar10ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = 'WideResNet%s-%s' %(args.depth, args.width)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    model = WideResNet(args.depth, args.width, num_classes=10)
    model.compile(loss='categorical_crossentropy',
            optimizer=SGD(lr=0.1, momentum=0.9),
            metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    datagen = Cifar10ImageDataGenerator(args)

    x_test = datagen.standardize(x_test)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    callbacks = [
        ModelCheckpoint('models/%s/model.hdf5'%args.name, verbose=1, save_best_only=True),
        CSVLogger('models/%s/log.csv'%args.name),
        CosineAnnealingScheduler(T_max=args.epochs, eta_max=0.05, eta_min=4e-4)
    ]

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        steps_per_epoch=len(x_train)//args.batch_size,
                        validation_data=(x_test, y_test),
                        epochs=args.epochs, verbose=1,
                        callbacks=callbacks)

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    main()

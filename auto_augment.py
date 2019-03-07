import random
import numpy as np
import scipy
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps


operations = {
    'ShearX': lambda img, prob, magnitude: shear_x(img, prob, magnitude),
    'ShearY': lambda img, prob, magnitude: shear_y(img, prob, magnitude),
    'TranslateX': lambda img, prob, magnitude: translate_x(img, prob, magnitude),
    'TranslateY': lambda img, prob, magnitude: translate_y(img, prob, magnitude),
    'Rotate': lambda img, prob, magnitude: rotate(img, prob, magnitude),
    'AutoContrast': lambda img, prob, magnitude: auto_contrast(img, prob, magnitude),
    'Invert': lambda img, prob, magnitude: invert(img, prob, magnitude),
    'Equalize': lambda img, prob, magnitude: equalize(img, prob, magnitude),
    'Solarize': lambda img, prob, magnitude: solarize(img, prob, magnitude),
    'Posterize': lambda img, prob, magnitude: posterize(img, prob, magnitude),
    'Contrast': lambda img, prob, magnitude: contrast(img, prob, magnitude),
    'Color': lambda img, prob, magnitude: color(img, prob, magnitude),
    'Brightness': lambda img, prob, magnitude: brightness(img, prob, magnitude),
    'Sharpness': lambda img, prob, magnitude: sharpness(img, prob, magnitude),
    'Cutout': lambda img, prob, magnitude: cutout(img, prob, magnitude),
}


def apply_policy(img, policy):
    img = operations[policy[0]](img, policy[1], policy[2])
    img = operations[policy[3]](img, policy[4], policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    transform_matrix = np.array([[1, magnitude/10*random.uniform(-0.3, 0.3), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def shear_y(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    transform_matrix = np.array([[1, 0, 0],
                                 [magnitude/10*random.uniform(-0.3, 0.3), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_x(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, magnitude/10*img.shape[1]*random.uniform(-150/331, 150/331)],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def translate_y(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    transform_matrix = np.array([[1, 0, magnitude/10*img.shape[1]*random.uniform(-150/331, 150/331)],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def rotate(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    theta = np.deg2rad(magnitude/10*random.uniform(-30, 30))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    return img


def auto_contrast(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageOps.autocontrast(img)
    img = np.array(img)
    return img


def invert(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageOps.invert(img)
    img = np.array(img)
    return img


def equalize(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageOps.equalize(img)
    img = np.array(img)
    return img


def solarize(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageOps.solarize(img, magnitude/10*random.uniform(0, 256))
    img = np.array(img)
    return img


def posterize(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageOps.posterize(img, round(magnitude/10*random.uniform(4, 8)))
    img = np.array(img)
    return img


def contrast(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageEnhance.Contrast(img).enhance(magnitude/10*random.uniform(0.1, 1.9))
    img = np.array(img)
    return img


def color(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(magnitude/10*random.uniform(0.1, 1.9))
    img = np.array(img)
    return img


def brightness(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageEnhance.Brightness(img).enhance(magnitude/10*random.uniform(0.1, 1.9))
    img = np.array(img)
    return img


def sharpness(img, prob=1.0, magnitude=10):
    if random.random() > prob:
        return img

    img = Image.fromarray(img)
    img = ImageEnhance.Sharpness(img).enhance(magnitude/10*random.uniform(0.1, 1.9))
    img = np.array(img)
    return img


def cutout(org_img, prob=1.0, magnitude=None):
    if random.random() > prob:
        return img

    img = np.copy(org_img)
    mask_val = img.mean()

    if magnitude is None:
        mask_size = 16
    else:
        mask_size = int(round(magnitude/10*img.shape[0]*random.uniform(0, 60/331)))
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    if top < 0:
        top = 0
    if left < 0:
        left = 0

    img[top:bottom, left:right, :].fill(mask_val)

    return img


def main():
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    img = x_train[0]
    for key, op in zip(operations.keys(), operations.values()):
        print(key)
        dst = op(img, 1.0, 10)
        plt.imshow(dst)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()

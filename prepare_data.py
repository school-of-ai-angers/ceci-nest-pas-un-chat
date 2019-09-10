import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

VALIDATION_RATIO = 0.3
TEST_RATIO = 0.1
TEST_CLASSES = ['cat', 'dog']
TARGET_SIZE = (128, 128)


def split_sets(elements, ratios):
    """
    Split a list into sets (randomly, but deterministically)
    :param elements: a list with N elements
    :param ratios: a list with K-1 floats that must sum less or equal to 1
    :returns: a list with K lists of elements
    """
    quantities = np.floor(np.array(ratios) * len(elements)).astype('int')
    np.random.seed(17)
    np.random.shuffle(elements)
    result = []
    last_index = 0
    for quantity in quantities:
        result.append(elements[last_index:last_index+quantity])
        last_index += quantity
    result.append(elements[last_index:])
    return result


if not os.path.exists('data/natural-images.zip'):
    raise Exception(
        'Please download the dataset from https://www.kaggle.com/prasunroy/natural-images and save it as data/natural-images.zip\n' +
        'The Kaggle website will require you to create an account, sorry about that...')

if not os.path.exists('data/natural_images'):
    print('=== Extract ZIP ===')
    ZipFile('data/natural-images.zip').extractall('data')

if not os.path.exists('data/training'):
    print('=== Prepare training, validation and test sets ===')

    # Load all files and split into sets
    files_by_destination = {}
    for class_dir in os.scandir('data/natural_images'):
        class_name = os.path.basename(class_dir.path)
        file_names = sorted(
            img_file.path for img_file in os.scandir(class_dir.path))
        if class_name in TEST_CLASSES:
            # train/validation/test
            a, b, c = split_sets(file_names, [VALIDATION_RATIO, TEST_RATIO])
            files_by_destination[f'data/validation/{class_name}'] = a
            files_by_destination[f'data/test/{class_name}'] = b
            files_by_destination[f'data/training/{class_name}'] = c
        else:
            # train/validation
            a, b = split_sets(file_names, [VALIDATION_RATIO])
            files_by_destination[f'data/validation/{class_name}'] = a
            files_by_destination[f'data/training/{class_name}'] = b

    for destination, files in files_by_destination.items():
        print(f'- Preprocess {len(files)} images to {destination}')
        if not os.path.exists(destination):
            os.makedirs(destination)
        for a_file in files:
            img = Image.open(a_file)

            # Fit image
            k_w = TARGET_SIZE[0] / img.width
            k_h = TARGET_SIZE[1] / img.height
            k = min(k_w, k_h)
            w = round(img.width * k)
            h = round(img.height * k)
            img = img.resize((w, h), Image.BICUBIC)
            assert img.width == TARGET_SIZE[0] or img.height == TARGET_SIZE[1]

            # Pad image with repeating pixels
            # (using dark pixels disturb the CNN)
            array = np.array(img)
            delta_w = TARGET_SIZE[0] - img.width
            delta_h = TARGET_SIZE[1] - img.height
            padding = (
                (delta_h // 2, delta_h - delta_h // 2),
                (delta_w // 2, delta_w - delta_w // 2),
                (0, 0)
            )
            array = np.pad(array, padding, 'edge')
            img = Image.fromarray(array)
            assert img.width == TARGET_SIZE[0] and img.height == TARGET_SIZE[1]

            # Save image
            img.save(f'{destination}/{os.path.basename(a_file)}')

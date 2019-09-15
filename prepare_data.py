import os
from zipfile import ZipFile
from PIL import Image
import numpy as np
from collections import defaultdict

VALIDATION_RATIO = 0.3
TEST_RATIO = 0.1
TARGET_SIZE = (224, 224)


def split_sets(elements, ratios):
    """
    Split a list into sets (randomly, but deterministically)
    :param elements: a list with N elements
    :param ratios: a list with K-1 floats that must sum less or equal to 1
    :returns: a list with K lists of elements
    """


if not os.path.exists('data/dogs-vs-cats.zip'):
    raise Exception(
        'Please download the dataset from https://www.kaggle.com/c/dogs-vs-cats/data and save it as data/dogs-vs-cats.zip\n' +
        'The Kaggle website will require you to create an account, sorry about that...')

if not os.path.exists('data/dogs-vs-cats/train'):
    print('=== Extract ZIP ===')
    ZipFile('data/dogs-vs-cats.zip').extractall('data/dogs-vs-cats')
    ZipFile('data/dogs-vs-cats/train.zip').extractall('data/dogs-vs-cats')

if not os.path.exists('data/training'):
    print('=== Prepare training, validation and test sets ===')

    # Load files by class
    files_by_class = defaultdict(list)
    for img_file in os.scandir('data/dogs-vs-cats/train'):
        files_by_class[img_file.name.split('.')[0]].append(img_file.path)

    # Load all files and split into sets
    files_by_destination = {}
    for class_name, file_names in files_by_class.items():
        file_names = sorted(file_names)
        num_validation = int(VALIDATION_RATIO * len(file_names))
        num_test = int(TEST_RATIO * len(file_names))
        np.random.seed(17)
        np.random.shuffle(file_names)
        files_by_destination[f'data/validation/{class_name}'] = file_names[:num_validation]
        files_by_destination[f'data/test/{class_name}'] = file_names[num_validation:num_validation+num_test]
        files_by_destination[f'data/training/{class_name}'] = file_names[num_validation+num_test:]

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

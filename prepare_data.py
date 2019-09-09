import os
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import numpy as np

VALIDATION_RATIO = 0.2
VALIDATION_CLASSES = ['cat', 'dog']
TARGET_SIZE = (128, 128)

if not os.path.exists('data/natural-images.zip'):
    raise Exception(
        'Please download the dataset from https://www.kaggle.com/prasunroy/natural-images and save it as data/natural-images.zip\n' +
        'The Kaggle website will require you to create an account, sorry about that...')

if not os.path.exists('data/natural_images'):
    print('=== Extract ZIP ===')
    ZipFile('data/natural-images.zip').extractall('data')

if not os.path.exists('data/training'):
    print('=== Prepare validation and training sets ===')

    # Load all files (sorted by name)
    files_by_class = {}
    for class_dir in os.scandir('data/natural_images'):
        files_by_class[os.path.basename(class_dir.path)] = sorted([
            img_file.path for img_file in os.scandir(class_dir.path)
        ])

    # Split training/validation
    training_by_class = {}
    validation_by_class = {}
    for class_name, files in files_by_class.items():
        if class_name not in VALIDATION_CLASSES:
            training_by_class[class_name] = files
        else:
            training_by_class[class_name], validation_by_class[class_name] = train_test_split(files, test_size=VALIDATION_RATIO, random_state=17, shuffle=True)
    
    # Crop and copy files
    def treat_files(files_by_class, subset):
        for class_name, files in files_by_class.items():
            dir_destination = f'data/{subset}/{class_name}'
            print(f'- Copy preprocessed images to {dir_destination}')
            if not os.path.exists(dir_destination):
                os.makedirs(dir_destination)
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
                img.save(f'{dir_destination}/{os.path.basename(a_file)}')

    treat_files(training_by_class, 'training')
    treat_files(validation_by_class, 'validation')
    
    
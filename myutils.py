import tensorflow as tf
import holoviews as hv
import numpy as np
import pandas as pd


def plot_img(x, y, classes):
    """
    Display a single image nicely
    :param x: an array shaped (width, height, 3)
    :param y: an array shaped (n)
    :param classes: a list of string shaped (n)
    :returns: hv.RGB
    """
    label = f'{classes[np.argmax(y)]} at {np.max(y):.0%}'
    return hv.RGB(x/255, label=label).opts(xaxis='bare', yaxis='bare', width=x.shape[0], height=x.shape[1])


def plot_imgs(X, Y, classes, cols=4):
    """
    Display many images nicely
    :param x: an array shaped (m, width, height, 3)
    :param y: an array shaped (m, n)
    :param classes: a list of string shaped (n)
    :param cols: int
    :returns: hv.Layout
    """
    layout = None
    for x, y in zip(X, Y):
        img = plot_img(x, y, classes)
        layout = img if layout is None else layout + img
    return layout.cols(cols)


def collect_images_from_directory(directory, target_size):
    """
    Collect all images from a directory
    :param directory: string the target directory
    :param target_size: tuple (width, height)
    :returns: two arrays, both shaped (samples, width, height, 3)
    """
    base_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    generator = base_generator.flow_from_directory(
        directory, target_size=target_size)
    Xs = []
    Ys = []
    for i in range(len(generator)):
        X, Y = generator.next()
        Xs.append(X)
        Ys.append(Y)
    return np.concatenate(Xs), np.concatenate(Ys)


def plot_train_history(history):
    """
    Plot the train and validation loss and accuracy
    :param history: the output of tf.keras.Model.fit
    :returns: hv.Layout
    """
    ds = hv.Dataset(pd.DataFrame(history.history), kdims=[('index', 'epoch')])
    loss = ds.to(hv.Curve, vdims=['loss'], label='Train') * \
        ds.to(hv.Curve, vdims=['val_loss'], label='Validation')
    accuracy = ds.to(hv.Curve, vdims=['accuracy'], label='Train') * \
        ds.to(hv.Curve, vdims=['val_accuracy'], label='Validation')
    best_epoch = hv.VLine(np.argmin(history.history['val_loss'])).opts(
        color='black', line_dash='dashed', line_width=1)
    return (loss + accuracy).opts(hv.opts.Curve(width=400, height=400, tools=['hover'])) * best_epoch


def get_weight_by_class(Y_ids):
    """
    Return a mapping from class id to recommended weight so that they'll
    contribute equally to the average loss of an epoch
    :param Y_ids: an array shaped (n)
    :returns: dict from int to float
    """
    class_id, class_count = np.unique(Y_ids, return_counts=True)
    avg_num = len(Y_ids) / len(class_id)
    return {k: avg_num / v for k, v in zip(class_id, class_count)}


def get_predict_df(val_Y, val_Y_pred, classes):
    """
    Return a nice DataFrame with the results of the validation
    :param val_Y: an array shaped (m, n)
    :param val_Y_pred: an array shaped (m, n)
    :param classes: a list of string shaped (n)
    :returns: pd.DataFrame
    """
    df = pd.DataFrame(val_Y_pred, columns=[f'class_{c}' for c in classes])
    df['real'] = pd.Categorical(np.argmax(val_Y, axis=1), range(
        len(classes))).rename_categories(classes)
    df['predicted'] = pd.Categorical(np.argmax(val_Y_pred, axis=1), range(
        len(classes))).rename_categories(classes)
    df['confidence'] = np.max(val_Y_pred, axis=1)
    df['loss'] = tf.keras.losses.categorical_crossentropy(val_Y, val_Y_pred)
    df['correct'] = df['real'] == df['predicted']
    return df


def plot_best_and_worst(df, X, Y_pred, classes, n=8):
    """
    Plot the best and worst cases
    :param df: pd.DataFrame returned by get_predict_df()
    :param X: an array shaped (m, width, height, 3)
    :param Y_pred: an array shaped (m, n)
    :param classes: a list of string shaped (n)
    :param n: int
    :returns: hv.Layout
    """
    best_5 = df.nsmallest(n, 'loss').index.values
    worst_5 = df.nlargest(n, 'loss').index.values
    return (plot_imgs(X[best_5], Y_pred[best_5], classes) +
            plot_imgs(X[worst_5], Y_pred[worst_5], classes)).opts(title=f'Best and worst {n}').cols(4)


def plot_confidence_distribution(df):
    """
    Plot the distribution of confidence when right and wrong
    :param df: pd.DataFrame returned by get_predict_df()
    :returns: hv.Layout
    """
    def get_histogram(df, title):
        curve = hv.Curve(df, ['index', 'real'], ['confidence'])
        return hv.operation.histogram(curve, groupby=['real'], num_bins=10, bin_range=(0.5, 1), normed=False) \
            .opts(width=800, height=400, title=title) \
            .opts(hv.opts.Histogram(alpha=0.5))[['cat', 'dog']]

    return (
        get_histogram(df[df['correct']], 'Distribution of confidence when right by ground truth') +
        get_histogram(df[~df['correct']],
                      'Distribution of confidence when wrong by ground truth')
    ).cols(1)

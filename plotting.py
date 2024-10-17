import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(spectrograms, labels, class_names, cols=3, rows=3, save=False):
    plt.figure(figsize=(7,8))
    for i in range(cols*rows):
        spectrogram = np.log(np.squeeze(spectrograms[i].numpy(), axis=-1).T + np.finfo(float).eps)
        plt.subplot(rows, cols, i+1)
        plt.imshow(spectrogram)
        plt.title(class_names[labels[i]])
    plt.show()
    if save:
        plt.savefig('spectrograms.png')

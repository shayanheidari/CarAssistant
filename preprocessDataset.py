from librosa import example
import numpy as np
import tensorflow as tf
from typing import Any, Tuple
import os
import matplotlib.pyplot as plt


class PreprocessDataset():
    def __init__(self, ds_path: str | os.PathLike) -> None:
        self.ds_path = ds_path
        self.class_names = [x[1] for x in os.walk(ds_path)][0]
        self.sr = 44100
    
    def load_audio_dataset(self, batch_size: int
                           ) -> Tuple[tf.data.Dataset, ...]:
        train_ds, valid_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=self.ds_path,
            batch_size=batch_size,
            seed=0,
            validation_split=0.2,
            output_sequence_length=self.sr * 2,
            subset='both')
        train_ds = train_ds.map(lambda audio, labels: (tf.squeeze(audio,
            axis=-1), labels), tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(lambda audio, labels: (tf.squeeze(audio,
            axis=-1), labels), tf.data.AUTOTUNE)
        test_ds = valid_ds.shard(num_shards=2, index=0)
        valid_ds = valid_ds.shard(num_shards=2, index=1)
        return train_ds, valid_ds, test_ds

    def add_white_noise(self, audio, label) -> Tuple[tf.Tensor, ...]:
        noise_factor = 0.007
        audio = tf.cast(audio, tf.float32)
        noise = tf.random.normal(
            shape=tf.shape(audio), mean=0.0, stddev=1.0, dtype=tf.float32
        )
        augmented_audio = audio + noise_factor * noise
        audio = tf.clip_by_value(augmented_audio, -1.0, 1.0)
        return audio, label

    def preprocess_to_spectrogram(self, audio, label) -> Tuple[tf.Tensor, ...]:
        spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram, label

def main() -> None:
    ds = os.path.join('keyword')
    test = PreprocessDataset(ds)
    train_ds, _, _= test.load_audio_dataset(20)
    # train_white_ds = train_ds.map(test.add_white_noise)
    train_spec_ds = train_ds.map(test.preprocess_to_spectrogram, tf.data.AUTOTUNE)
    for example_spectrograms , label in train_spec_ds.take(1):
        break
    for i in example_spectrograms:
        plt.imshow(i)
        plt.show()


if __name__ == '__main__':
    main()

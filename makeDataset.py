import sounddevice as sd
from tqdm import tqdm
from scipy.io.wavfile import write
import os


class MakeDataset():
    def __init__(self, ds_path: str | os.PathLike, n_sample: int) -> None:
        self.ds_path = ds_path
        self.n_sample = n_sample
        self.sr = 44100

    def record_save_audio(self, class_name: str = 'sample', duration: int = 2) -> None:
        input(f'You are recording audio for class {class_name!r} '
              f'press Enter to start: ')
        os.system('cls' if os.name == 'nt' else 'clear')
        for sample in tqdm(range(self.n_sample)):
            if not os.path.exists(class_path := (self.ds_path + "/" + class_name)):
                os.mkdir(class_path)
            recorded_sound = sd.rec(int(duration * self.sr), samplerate=self.sr, dtype='int16', channels=1)
            sd.wait()
            save_path = os.path.join(self.ds_path, class_name, class_name)
            write(save_path + '_' + str(sample) + '.wav', self.sr, recorded_sound)
            input(f'{self.n_sample - sample} sample to record for class {class_name},'
                  f' press Enter to continue: ')
            os.system('cls' if os.name == 'nt' else 'clear')

    def record_save_background(self, class_name: str = 'background', duration: int = 2):
        input(f'You are recording audio for class {class_name!r} '
              f'press Enter to start: ')
        os.system('cls' if os.name == 'nt' else 'clear')
        for sample in tqdm(range(self.n_sample)):
            if not os.path.exists(class_path := (self.ds_path + "/" + class_name)):
                os.mkdir(class_path)
            recorded_sound = sd.rec(int(duration * self.sr), samplerate=self.sr, dtype='int16', channels=1)
            sd.wait()
            save_path = os.path.join(self.ds_path, class_name, class_name)
            write(save_path + '_' + str(sample) + '.wav', self.sr, recorded_sound)
            print(f'{self.n_sample - sample} sample to record, press Enter to continue: ')
            os.system('cls' if os.name == 'nt' else 'clear')


def main() -> None:
    dataset_path = os.path.join('datasets/keyword')
    keyword_ds = MakeDataset(ds_path=dataset_path, n_sample=10)
    keyword_ds.record_save_audio(class_name='machine')
    keyword_ds.record_save_background()


if __name__ == '__main__':
    main()

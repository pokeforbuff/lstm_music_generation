import os
import regex as re
import numpy as np


class MusicHelper:

    def __init__(self):
        self.cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.char2idx = None
        self.idx2char = None
        self.songs, self.vocab_size = self.load_training_data()

    def get_cwd(self):
        return os.path.abspath(os.path.join(self.cwd, os.pardir))

    def load_training_data(self):
        with open(os.path.join(self.cwd, "data", "irish.abc"), "r") as f:
            text = f.read()
        songs = self.extract_song_snippet(text)
        songs = "\n\n".join(songs)
        vocab = sorted(set(songs))
        vocab_size = len(vocab)
        self.char2idx = {u: i for i, u in enumerate(vocab)}
        self.idx2char = np.array(vocab)
        songs = np.array([self.encode_music_note(char) for char in songs])
        return songs, vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_training_batch(self, seq_length, batch_size):
        n = self.songs.shape[0] - 1
        idx = np.random.choice(n - seq_length, batch_size)
        input_batch = [self.songs[index: index + seq_length] for index in idx]
        output_batch = [self.songs[index + 1: index + 1 + seq_length] for index in idx]
        x_batch = np.reshape(input_batch, [batch_size, seq_length])
        y_batch = np.reshape(output_batch, [batch_size, seq_length])
        return x_batch, y_batch

    def encode_music_note(self, note):
        return self.char2idx[note]

    def decode_music_note(self, note):
        return self.idx2char[note]

    def save_generated_songs(self, generated_abc_text):
        songs = self.extract_song_snippet(generated_abc_text)
        for i, song in enumerate(songs):
            waveform = self.save_song(song)
            if waveform:
                print("Generated song", i)

    def get_number_of_generated_songs(self):
        dir_path = os.path.join(self.cwd, 'generated_songs')
        count = 0
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count

    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
        songs = [song[1] for song in search_results]
        return songs

    def convert_song_to_abc(self, song):
        song_id = self.get_number_of_generated_songs()
        filename = "song_{}".format(song_id)
        save_name = "{}.abc".format(filename)
        with open(os.path.join(self.cwd, 'generated_songs', save_name), "w") as f:
            f.write(song)
        return filename

    def convert_abc_to_wav(self, abc_file):
        path_to_tool = os.path.join(self.cwd, 'bin', 'abc2wav.sh')
        path_to_abc_file = os.path.join(self.cwd, 'generated_songs', abc_file)
        cmd = "{} {} {}".format("sh", path_to_tool, path_to_abc_file)
        return os.system(cmd)

    def save_song(self, song):
        abc_file = self.convert_song_to_abc(song)
        wav_file = self.convert_abc_to_wav(abc_file + '.abc')
        return wav_file == 0


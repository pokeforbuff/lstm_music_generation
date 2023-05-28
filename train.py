from data.music_helper import MusicHelper
from models import LSTM

num_training_iterations = 10000
learning_rate = 5e-3
seq_length = 300

batch_size = 32
embedding_dim = 256
rnn_units = 1024


model = LSTM(
    embedding_dim,
    rnn_units,
    batch_size
)

loss_history = model.train(
    num_training_iterations,
    learning_rate,
    seq_length
)

print(loss_history[-1])

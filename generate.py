from models import LSTM

start_string = "X"
generation_length = 1000

batch_size = 1
embedding_dim = 256
rnn_units = 1024


model = LSTM(
    embedding_dim,
    rnn_units,
    batch_size
)

model.generate(
    start_string,
    generation_length
)

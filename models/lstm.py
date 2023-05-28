import tensorflow as tf

import os
from data.music_helper import MusicHelper


class LSTM:
    def __init__(
            self,
            embedding_dim,
            rnn_units,
            batch_size
    ):
        self.checkpoint_dir, self.checkpoint_prefix = self.init_checkpoint_manager()
        self.music_helper = MusicHelper()
        self.batch_size = batch_size

        self.model = self.init_model(
            self.music_helper.get_vocab_size(),
            embedding_dim,
            rnn_units
        )

    def init_model(self, vocab_size, embedding_dim, rnn_units):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                batch_input_shape=[self.batch_size, None]
            ),
            tf.keras.layers.LSTM(
                rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                recurrent_activation='sigmoid',
                stateful=True,
            ),
            tf.keras.layers.Dense(
                vocab_size
            )
        ])
        return model

    def init_checkpoint_manager(self):
        checkpoint_dir = './training_checkpoints'
        return checkpoint_dir, os.path.join(checkpoint_dir, "checkpoint")

    def get_loss(self, labels, logits):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x)
            loss = self.get_loss(y, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, num_training_iterations, learning_rate, seq_length):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss_history = []
        for iteration in range(num_training_iterations):
            print(f"Train Iteration {iteration}")
            x_batch, y_batch = self.music_helper.get_training_batch(seq_length, self.batch_size)
            loss = self.train_step(x_batch, y_batch)
            loss_history.append(loss.numpy().mean())
        self.model.save_weights(self.checkpoint_prefix)
        return loss_history

    def generate(self, start_string, generation_length=1000):
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))

        input_eval = [self.music_helper.encode_music_note(s) for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        self.model.reset_states()
        for i in range(generation_length):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.music_helper.decode_music_note(predicted_id))
        generated_text = start_string + ''.join(text_generated)
        self.music_helper.save_generated_songs(generated_text)

from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense # type: ignore

def build_model(num_chars, latent_dim):
    encoder_inputs = Input(shape = (None, num_chars))

    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape = (None, num_chars))

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense = Dense(num_chars, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
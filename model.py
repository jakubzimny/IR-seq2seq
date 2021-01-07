from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import optimizers, metrics, backend as K

def calculate_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def calculate_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def s2s_model(hidden_size, input_chars_count, input_target_count):
    encoder_inputs = Input(shape=(None, input_chars_count))
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.2, return_sequences=True, return_state=False)
    encoder_outputs = encoder_lstm(encoder_inputs)
    encoder_lstm = LSTM(hidden_size, recurrent_dropout=0.2, return_sequences=False, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_outputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, input_target_count))
    decoder_lstm = LSTM(hidden_size, recurrent_dropout=0.2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_softmax = Dense(input_target_count, activation='softmax')
    decoder_outputs = decoder_softmax(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])

    return model
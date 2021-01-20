import re
import os
import numpy as np

from unidecode import unidecode
from typing import List, Tuple
from model import f1_score
from onehot import OneHotEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

SOS = '\t'
EOS = '*' 
CHARSET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

def load_s2s_model(path, hidden_size, double_lstm = False):
    model = load_model(path, custom_objects={'f1_score': f1_score})
    if double_lstm:
        pass
    else:
        encoder_inputs = model.input[0]
        encoder_lstm1 = model.get_layer('e_lstm1')
        encoder_lstm2 = model.get_layer('e_lstm2')
        encoder_outputs = encoder_lstm1(encoder_inputs)

        _, state_h, state_c = encoder_lstm2(encoder_outputs)
        encoder_states = [state_h, state_c]
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

        decoder_inputs = model.input[1]
        decoder_state_input_h = Input(shape=(hidden_size,))
        decoder_state_input_c = Input(shape=(hidden_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.get_layer('d_lstm')
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_softmax = model.get_layer('d_softmax')
        decoder_outputs = decoder_softmax(decoder_outputs)
        decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs,
                            outputs=[decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

def decode_sequences(inputs, input_oh_encoder, target_oh_encoder, max_length, encoder, decoder, nb_of_examples, is_reversed):
    input_tokens = []
    indices = range(nb_of_examples)
    for index in indices:
        input_tokens.append(inputs[index])

    input_sequences = next(get_batch_generator(tokens=input_tokens,
                                          max_length=max_length,
                                          oh_encoder=input_oh_encoder,
                                          batch_size=nb_of_examples,
                                          is_reversed=is_reversed))

    states = encoder.predict(input_sequences)
    target_sequences = np.zeros((nb_of_examples, 1, target_oh_encoder.get_charset_size()))
    target_sequences[:, 0, target_oh_encoder.get_index_of_char(SOS)] = 1.0

    decoded_tokens = ['' for _ in range(nb_of_examples)]
    for _ in range(max_length):
        char_probs, h, c = decoder.predict([target_sequences] + states)
        target_sequences = np.zeros((nb_of_examples, 1, target_oh_encoder.get_charset_size()))
        sampled_chars = []
        for i in range(nb_of_examples):
            next_index, next_char = target_oh_encoder.decode_one_hot(char_probs[i])
            decoded_tokens[i] += next_char
            sampled_chars.append(next_char)
            target_sequences[i, 0, next_index] = 1.0
        stop_char = set(sampled_chars)
        if len(stop_char) == 1 and stop_char.pop() == EOS:
            break
        states = [h, c]

    input_tokens   = [re.sub('[%s]' % EOS, '', token)
                      for token in input_tokens]
    decoded_tokens = [re.sub('[%s]' % EOS, '', token)
                      for token in decoded_tokens]
    return input_tokens,  decoded_tokens


def load_text(data_dir: str, subfolder: str ='train') -> str:
    text = ''
    file_list = os.listdir(os.path.join(data_dir, subfolder))
    for file_path in file_list:
        with open(os.path.join(data_dir, subfolder, file_path)) as f:    
            text += unidecode(f.read()) + ' '
    return text

def tokenize(text: str) -> List:
    chars_to_remove = list(r'#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c')
    tokens = re.split('[-\n ]', text)
    for token in tokens:
        for char in chars_to_remove:
            token.replace(char, '')
    return tokens
    
def add_speling_erors(token: str, error_rate: float):
    if len(token) < 3:
        return token
    rand = np.random.rand()
    chance = error_rate / 4.0
    random_idx = np.random.randint(len(token))
    if rand < chance:  # replace
        token = token[:random_idx] + np.random.choice(CHARSET) + token[random_idx + 1:]
    elif chance < rand < 2 * chance:  # transpose
        random_idx = np.random.randint(len(token) - 1)
        token = token[:random_idx] + token[random_idx + 1] + token[random_idx] + token[random_idx + 2:]
    elif chance * 2 < rand < 3 * chance:  # add
        token = token[:random_idx] + np.random.choice(CHARSET) + token[random_idx:]
    elif chance * 3 < rand < 4 * chance:  # delete
       token = token[:random_idx] + token[random_idx + 1:]
    return token

def get_padded_token(token: str, max_length: int) -> str:
    padded_token = token
    for _ in range(len(token), max_length):
        padded_token += EOS
    return padded_token

def prepare_word_tokens(tokens: List, max_length: int, error_rate: float = 0.3) -> Tuple:
    encoder_tokens = []
    decoder_tokens = []
    target_tokens = []
    for token in tokens:
        encoder_token = add_speling_erors(token, error_rate=error_rate)
        encoder_tokens.append(get_padded_token(encoder_token, max_length))
        decoder_token = SOS + token
        decoder_tokens.append(get_padded_token(decoder_token, max_length)) 
        target_token = decoder_token[1:]
        target_tokens.append(get_padded_token(target_token, max_length))
    return encoder_tokens, decoder_tokens, target_tokens

def get_token_generator(tokens: List, is_reversed: bool = False):
    while(True):
        for token in tokens:
            if is_reversed:
                token = token[::-1]
            yield token

def get_batch_generator(tokens: List, max_length: int, oh_encoder, batch_size: int = 128, is_reversed: bool = False):
    token_iterator = get_token_generator(tokens, is_reversed)
    batch = np.zeros((batch_size, max_length, oh_encoder.get_charset_size()), dtype=np.float32)
    while(True):
        for i in range(batch_size):
            token = next(token_iterator)
            batch[i] = oh_encoder.encode_one_hot(token, max_length)
        yield batch


def get_data_generator(encoder_iter, decoder_iter, target_iter):
    inputs = zip(encoder_iter, decoder_iter)
    while(True):
        encoder_input, decoder_input = next(inputs)
        target = next(target_iter)
        yield ([encoder_input, decoder_input], target)



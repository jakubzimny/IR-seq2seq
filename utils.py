import re
import os
import numpy as np

from unidecode import unidecode
from typing import List, Tuple

SOS = '\t'
EOS = '*' 
CHARSET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')

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

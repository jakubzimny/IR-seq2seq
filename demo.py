from onehot import OneHotEncoder
from tensorflow.keras.models import load_model

from utils import load_text, tokenize, get_padded_token, load_s2s_model, decode_sequences, prepare_word_tokens
from model import s2s_model

HIDDEN_SIZE = 512 
ERR_RATE = 0.8  
BATCH_SIZE = 256 
DATA_DIR = './data'

if __name__ == '__main__':
    # Prepare model
    encoder, decoder = load_s2s_model('test-no_reverse-hs-512_err-0.8_bs-256_e-30_drop-0.2.h5', HIDDEN_SIZE)

    text  = load_text(DATA_DIR)
    word_set = list(filter(None, set(tokenize(text))))
    max_word_len = max([len(token) for token in word_set]) + 2 
    train_enc_tokens, train_dec_tokens, _ = prepare_word_tokens(word_set, max_word_len, ERR_RATE)

    enc_charset = set(' '.join(train_enc_tokens))
    dec_charset = set(' '.join(train_dec_tokens))
    enc_oh = OneHotEncoder(enc_charset)
    dec_oh = OneHotEncoder(dec_charset)

    # Input decoding loop
    while True:
        sentence = input('\nEnter sentence to decode:\n')
        tokens = list(filter(None,tokenize(sentence)))
        nb_of_tokens = len(tokens)
        prepared_tokens = []
        for token in tokens:
            prepared_tokens.append(get_padded_token(token, max_word_len))
        input_tokens, decoded_tokens = decode_sequences(prepared_tokens, enc_oh,
                                                        dec_oh, max_word_len,
                                                        encoder, decoder,
                                                        nb_of_tokens, False)
        print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
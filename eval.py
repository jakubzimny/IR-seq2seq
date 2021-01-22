from onehot import OneHotEncoder
from tensorflow.keras.models import load_model

from utils import load_text, tokenize, get_padded_token, load_s2s_model, decode_sequences, prepare_word_tokens, get_batch_generator
from model import s2s_model

HIDDEN_SIZE = 512 
ERR_RATE = 0.2  
BATCH_SIZE = 256 
DATA_DIR = './data'

if __name__ == '__main__':
    encoder, decoder = load_s2s_model('test-no_reverse-hs-512_err-0.8_bs-256_e-100_drop-0.2.h5', HIDDEN_SIZE)

    text  = load_text(DATA_DIR)
    test_text = load_text(DATA_DIR, 'test')
    word_set = list(filter(None, set(tokenize(text))))
    test_word_set = list(filter(None, set(tokenize(test_text))))
    train_max_word_len = max([len(token) for token in word_set]) + 2 
    test_max_word_len = max([len(token) for token in test_word_set]) + 2
    train_enc_tokens, train_dec_tokens, _ = prepare_word_tokens(word_set, train_max_word_len, ERR_RATE)
    test_enc_tokens, test_dec_tokens, test_target_tokens = prepare_word_tokens(test_word_set, test_max_word_len, ERR_RATE)

    enc_charset = set(' '.join(train_enc_tokens))
    dec_charset = set(' '.join(train_dec_tokens))

    enc_oh = OneHotEncoder(enc_charset)
    dec_oh = OneHotEncoder(dec_charset)

    token_count = len(test_enc_tokens)
    right_guess_counter = 0
    counter = 0

    for enc_token, dec_token, target_token in zip(test_enc_tokens, test_enc_tokens, test_target_tokens):
        _, decoded_token = decode_sequences([enc_token], enc_oh,
                                            dec_oh, train_max_word_len,
                                            encoder, decoder,
                                            1, False)
        #print(f'Decoded: {decoded_token[0]} | Target: {target_token.rstrip("*")}')
        if decoded_token[0] == target_token.rstrip('*'):
            right_guess_counter += 1
        counter +=1
        if (counter % 10 == 0):
            print(f'{((counter/token_count) * 100):.3f}% finished')
    print(f'Accuracy: {right_guess_counter/token_count}')
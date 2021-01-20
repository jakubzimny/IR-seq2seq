import os
import numpy as np

from tensorflow.keras import callbacks
from utils import get_batch_generator, get_data_generator
from utils import load_text, tokenize, prepare_word_tokens
from onehot import OneHotEncoder
from model import s2s_model

HIDDEN_SIZE = 512  
ERR_RATE = 0.2  
EPOCHS = 15
BATCH_SIZE = 256  
DATA_DIR = './data'
R_DROPUT = 0.2

if __name__ == '__main__':
    train_text  = load_text(DATA_DIR)
    val_text = load_text(DATA_DIR, 'val')

    train_word_set = list(filter(None, set(tokenize(train_text))))
    val_word_set = list(filter(None, set(tokenize(val_text))))
    
    train_max_word_len = max([len(token) for token in train_word_set]) + 2 
    val_max_word_len = max([len(token) for token in val_word_set]) + 2

    train_encoder_tokens, train_decoder_tokens, train_target_tokens = prepare_word_tokens(train_word_set,
                                                                                          train_max_word_len,
                                                                                          error_rate=ERR_RATE)
    val_encoder_tokens, val_decoder_tokens, val_target_tokens = prepare_word_tokens(val_word_set,
                                                                                    val_max_word_len, 
                                                                                    error_rate=ERR_RATE)

    input_charset = set(' '.join(train_encoder_tokens))
    target_charset = set(' '.join(train_decoder_tokens))
    input_oh_encoder  = OneHotEncoder(input_charset)
    target_oh_encoder = OneHotEncoder(target_charset)

    train_steps = len(train_word_set) // BATCH_SIZE
    val_steps = len(val_word_set) // BATCH_SIZE

    model = s2s_model(HIDDEN_SIZE, len(input_charset), len(target_charset), R_DROPUT)
    print(model.summary())

    model_name = f'test-no_reverse-hs-{HIDDEN_SIZE}_err-{ERR_RATE}_bs-{BATCH_SIZE}_e-{EPOCHS}_drop-{R_DROPUT}'
    log_dir = f'tb_logs/{model_name}'
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
    train_encoder_batch = get_batch_generator(train_encoder_tokens, train_max_word_len, input_oh_encoder, BATCH_SIZE, False)
    train_decoder_batch = get_batch_generator(train_decoder_tokens, train_max_word_len, target_oh_encoder, BATCH_SIZE)
    train_target_batch  = get_batch_generator(train_target_tokens, train_max_word_len, target_oh_encoder, BATCH_SIZE)    

    val_encoder_batch = get_batch_generator(val_encoder_tokens, val_max_word_len, input_oh_encoder, BATCH_SIZE, False)
    val_decoder_batch = get_batch_generator(val_decoder_tokens, val_max_word_len, target_oh_encoder,BATCH_SIZE)
    val_target_batch  = get_batch_generator(val_target_tokens, val_max_word_len, target_oh_encoder, BATCH_SIZE)
    
    train_loader = get_data_generator(train_encoder_batch, train_decoder_batch, train_target_batch)
    val_loader = get_data_generator(val_encoder_batch, val_decoder_batch, val_target_batch)
    
    model.fit(train_loader,
              steps_per_epoch=train_steps,
              epochs=EPOCHS, verbose=1,
              validation_data=val_loader,
              validation_steps=val_steps,
              callbacks=[tensorboard_callback])    

    model.save(f'{model_name}.h5')
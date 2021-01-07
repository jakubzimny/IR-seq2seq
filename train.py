import os
import numpy as np

from tensorflow.keras import callbacks
from utils import CharacterTable, prepare_tokens
from utils import get_batch_generator, datagen, decode_sequences, generate_data
from utils import read_text, tokenize
from model import s2s_model

HIDDEN_SIZE = 512
ERR_RATE = 0.8
EPOCHS = 100
BATCH_SIZE = 128


if __name__ == '__main__':
    #error_rate = 0.8
    #nb_epochs = 100
    #train_batch_size = 128
   # val_batch_size = 256
    #sample_mode = 'argmax'

    reverse = True

    data_path = './data'
    train_books = ['nietzsche.txt', 'pride_and_prejudice.txt',
                'shakespeare.txt', 'war_and_peace.txt']
    val_books = ['wonderland.txt']
    # Prepare training data.
    text  = read_text(data_path, train_books)
    vocab = tokenize(text)
    vocab = list(filter(None, set(vocab)))
    
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = prepare_tokens(vocab, maxlen, error_rate=ERR_RATE)

    input_chars = set(' '.join(train_encoder))
    target_chars = set(' '.join(train_decoder))

    # Prepare validation data.
    text = read_text(data_path, val_books)
    #val_tokens = c
    val_tokens = list(filter(None, tokenize(text)))

    val_maxlen = max([len(token) for token in val_tokens]) + 2
    val_encoder, val_decoder, val_target = prepare_tokens(val_tokens, maxlen, error_rate=ERR_RATE)

    # Define training and evaluation configuration.
    input_ctable  = CharacterTable(input_chars)
    target_ctable = CharacterTable(target_chars)

    train_steps = len(vocab) // BATCH_SIZE
    val_steps = len(val_tokens) // BATCH_SIZE

    model = s2s_model(HIDDEN_SIZE, len(input_chars), len(target_chars))
    print(model.summary())

    log_dir = 'tb_logs/trash'
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #train_encoder, train_decoder, train_target = transform(vocab, maxlen, error_rate=error_rate)
        
    train_encoder_batch = get_batch_generator(train_encoder, maxlen, input_ctable, BATCH_SIZE, reverse)
    train_decoder_batch = get_batch_generator(train_decoder, maxlen, target_ctable, BATCH_SIZE)
    train_target_batch  = get_batch_generator(train_target, maxlen, target_ctable, BATCH_SIZE)    

    val_encoder_batch = get_batch_generator(val_encoder, maxlen, input_ctable, BATCH_SIZE, reverse)
    val_decoder_batch = get_batch_generator(val_decoder, maxlen, target_ctable,BATCH_SIZE)
    val_target_batch  = get_batch_generator(val_target, maxlen, target_ctable, BATCH_SIZE)
    
    train_loader = datagen(train_encoder_batch, train_decoder_batch, train_target_batch)
    val_loader = datagen(val_encoder_batch, val_decoder_batch, val_target_batch)
    
    model.fit(train_loader,
              steps_per_epoch=train_steps,
              epochs=EPOCHS, verbose=1,
              validation_data=val_loader,
              validation_steps=val_steps,
              callbacks=[tensorboard_callback])    

    model.save('model.h5')

    #Train and evaluate.
    # for epoch in range(nb_epochs):
    #     print('Main Epoch {:d}/{:d}'.format(epoch + 1, nb_epochs))
    
    #     train_encoder, train_decoder, train_target = transform(
    #         vocab, maxlen, error_rate=error_rate, shuffle=True)
        
    #     train_encoder_batch = batch(train_encoder, maxlen, input_ctable,
    #                                 train_batch_size, reverse)
    #     train_decoder_batch = batch(train_decoder, maxlen, target_ctable,
    #                                 train_batch_size)
    #     train_target_batch  = batch(train_target, maxlen, target_ctable,
    #                                 train_batch_size)    

    #     val_encoder_batch = batch(val_encoder, maxlen, input_ctable,
    #                               val_batch_size, reverse)
    #     val_decoder_batch = batch(val_decoder, maxlen, target_ctable,
    #                               val_batch_size)
    #     val_target_batch  = batch(val_target, maxlen, target_ctable,
    #                               val_batch_size)
    
    #     train_loader = datagen(train_encoder_batch,
    #                            train_decoder_batch, train_target_batch)
    #     val_loader = datagen(val_encoder_batch,
    #                          val_decoder_batch, val_target_batch)
    
    #     model.fit(train_loader,
    #               steps_per_epoch=train_steps,
    #               epochs=1, verbose=1,
    #               validation_data=val_loader,
    #               validation_steps=val_steps,
    #               initial_epoch=epoch,
    #               callbacks=[tensorboard_callback])

    #    # On epoch end - decode a batch of misspelled tokens from the
    #    # validation set to visualize speller performance.



    #     nb_tokens = 5
    #     input_tokens, target_tokens, decoded_tokens = decode_sequences(
    #         val_encoder, val_target, input_ctable, target_ctable,
    #         maxlen, reverse, encoder_model, decoder_model, nb_tokens,
    #         sample_mode=sample_mode, random=True)
        
    #     print('-')
    #     print('Input tokens:  ', input_tokens)
    #     print('Decoded tokens:', decoded_tokens)
    #     print('Target tokens: ', target_tokens)
    #     print('-')
        
    #     # Save the model at end of each epoch.
    #     model_file = '_'.join(['seq2seq', 'epoch', str(epoch + 1)]) + '.h5'
    #     save_dir = 'checkpoints'
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     save_path = os.path.join(save_dir, model_file)
    #     print('Saving full model to {:s}'.format(save_path))
    #     model.save(save_path)
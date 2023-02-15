import numpy as np 
import tensorflow as tf
import cv2
import pickle 
import matplotlib.pyplot as plt

from model import model 
from param import *

def generate_caption(image_path):
    image = cv2.imread(image_path,1)
    image = cv2.resize(image,(IMAGE_SIZE[0],IMAGE_SIZE[1]))
    image_batch = np.expand_dims(image,axis = 0)

    ker_mod = model()
    ker_mod.load_weights('./model_talk.h5')

    shape = (1,MAX_LENGTH)
    decoder_input_data = np.zeros(shape = shape,dtype = np.int32)
    decoder_input_data[0][0] = 2
    count_token = 0
    
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB,
        standardize=standardize,
        output_sequence_length=MAX_LENGTH)
    
    with open('tokenizer.pkl', 'rb') as f:
        dicts = pickle.load(f)
        tokenizer.from_config(dicts['config'])
        tokenizer.set_weights(dicts['weights'])
        
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)

    x_data = {
            'input_2':image_batch,
            'decoder_input':decoder_input_data 
            }
    mod = model()
    mod.load_weights('model_talk.h5')
    
    mod.summary()
    out = mod.predict(x_data)
    outs = np.argmax(out[0],axis = -1)
    
    text = " "
    for elem in outs:
        text += index_to_word(elem) + ' '
    print(text)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    generate_caption('./apples.jpg')

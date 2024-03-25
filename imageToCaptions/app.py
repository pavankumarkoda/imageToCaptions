from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import to_categorical, plot_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "second_model.h5"

vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Check http://127.0.0.1:5000/')

def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def cleaning(mapping) :
    for key, captions in mapping.items() :
        for i in range(len(captions)) :
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]','')
            caption = caption.replace('\s+',' ')
            caption = 'start ' + " ".join([word for word in caption.split() if len(word)>1]) + ' end'
            captions[i] = caption

captions_doc = load_doc("C:/Users/SANJAY KUMAR/Documents/Projects/Major/Dataset/Flickr8k_text/Flickr8k.token.txt")
mapping = {}
for line in tqdm(captions_doc.split('\n')) :
    t = line.split("\t")
    image_id, caption = t[0], t[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping :
        mapping[image_id] = []
    mapping[image_id].append(caption)
del mapping['']

cleaning(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)

def index_to_word(integer,tokenizer) :
    for word, index in tokenizer.word_index.items() :
        if index == integer :
            return word
    return None

def predict_caption(model, image, tokenizer, max_length) :
    in_text = 'start'
    for i in range(max_length) :
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq],max_length)
        nextword = model.predict([image,seq],verbose=0)
        nextword = np.argmax(nextword)
        word = index_to_word(nextword,tokenizer)
        if word is None :
            break
        in_text += " "+ word
        if word == 'end' :
            break
    return in_text

def model_predict(img_path, model):
    image = load_img(img_path,target_size=(224,224))
    #plt.imshow(image)
    img = img_to_array(image)
    img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
    img = preprocess_input(img)
    feature = vgg_model.predict(img,verbose=0)
    res = predict_caption(model,feature,tokenizer,max_length)
    return res

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        res = model_predict(file_path, model)
        return res[6:-4]
    return None


if __name__ == '__main__':
    app.run(debug=True)
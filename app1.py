import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)

# load model
model = load_model("garbage3.h5", compile=False)

# warm up model
dummy = np.zeros((1,32,32,3))
model.predict(dummy)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method == 'POST':

        f = request.files['image']
        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(32,32))
        x = image.img_to_array(img)
        x = x/255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)
        pred = np.argmax(pred)

        index=['cardboard','glass','metal','paper','plastic','trash']

        text = "The Classified waste is : " + index[pred]

        return text

    return render_template("base.html")


if __name__ == '__main__':
    app.run(debug=True)
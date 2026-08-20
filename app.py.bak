import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
app=Flask(__name__)

model=load_model(r"garbage3.h5",compile=False)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("base.html")

@app.route('/predict',methods=['GET','POST'])


def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['cardboard', 'glass', 'metal', 'paper', 'plastic','trash']
        text="The Classified waste is : " +str(index[pred[0]])
    return text



if __name__=='__main__':
    app.run()
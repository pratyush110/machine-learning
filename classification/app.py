from flask import Flask, render_template,request
import cv2
import numpy as np
import keras.models
import re
import base64
import sys
import os
sys.path.append(os.path.abspath("./model"))
from load import *
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
        imgData1 = imgData1.decode("utf-8")
        imgstr = re.search(r'base64,(.*)',imgData1).group(1)
        #print(imgstr)
        imgstr_64 = base64.b64decode(imgstr)
        with open('output/output.png','wb') as output:
                output.write(imgstr_64)


@app.route('/')
def index():
        #initModel()
        #render out pre-built HTML file right on the index page
        return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
        #whenever the predict method is called, we're going
        #to input the user drawn character as an image into the model
        #perform inference, and return the classification
        #get the raw data format of the image
        imgData = request.get_data()
        #print(imgData)
        #encode it into a suitable format
        convertImage(imgData)
        print("debug")
        #read the image into memory
        x = cv2.imread('output/output.png',0)
        #compute a bit-wise inversion so black becomes white and vice versa
        x = np.invert(x)
        #make it the right size
        x = cv2.resize(x,(28,28))
        #imshow(x)
        #convert to a 4D tensor to feed into our model
        x = x.reshape(1,28,28,1)
        print("debug2")
        #in our computation graph
        with graph.as_default():
                #perform the prediction
                out = model.predict(x)
                #print(out)
                print(np.argmax(out,axis=1))
                print("debug3")
                #convert the response to a string
                response = np.array_str(np.argmax(out,axis=1))
                return response


if __name__ == "__main__":
        #decide what port to run the app in
        port = int(os.environ.get('PORT', 5000))
        #run the app locally on the givn port
        app.run(host='0.0.0.0', port=port)
        #optional if we want to run in debugging mode
        #app.run(debug=False)

from flask import Flask, render_template, redirect, request
import os
import tensorflow as tf
import numpy as np
import cv2 #To process images, say to reduce image quality to change color scheme and so on.


app = Flask(__name__)

@app.route('/')
def handleHome():
    return render_template('index.html',pred = pred)

@app.route('/uploadimage',methods=['GET','POST'])
def handleUploadImage():
    if request.method =='POST':
        if request.files:
            imageFile = request.files['image']
            absPath = os.path.join(app.config['IMAGE_UPLOAD'],imageFile.filename)
            imageFile.save(absPath)
            imgArr = cv2.imread(absPath,cv2.IMREAD_GRAYSCALE)
            resizedImg = cv2.resize(imgArr,(28,28))
            pred = model.predict([[resizedImg]])
            predNum = np.argmax(pred)
            print(predNum)
    return render_template('index.html',pred = predNum )

if __name__ == "__main__":
    
    pred = -1
    app.config['IMAGE_UPLOAD'] = r'D:\NumberClassifier\templates\testImages' #Change the path according to your directory structure

    (x_test,y_test),(x_train,y_train) = tf.keras.datasets.mnist.load_data()  #I used mnist data set to train my model. ..
    x_test = tf.keras.utils.normalize(x_test)                                #..Data set contains 60k, 28px x 28px images of numbers
    x_train = tf.keras.utils.normalize(x_train)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))             #Two hidden layers of 128 neurons each
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))             #ReLU(Rectified Linear Unit) is an activation..
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))           #..function like Sigmoid, Tanh, Softmax ets
    
    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
    )
    
    model.fit(x_test,y_test,epochs=3)
    
    app.run(debug=True)
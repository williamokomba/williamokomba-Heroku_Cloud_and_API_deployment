#model deployment

import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle
import os
#creating an app
app = Flask(__name__, template_folder= os.path.join('templates')) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

#creating router
@app.route('/', methods=['Get', 'Post'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 3)
    
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

    
     
if __name__ == "__main__":
    app.run(port= 5000, debug=True)
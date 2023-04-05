import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

# create an object of the class Flask

app = Flask(__name__)
model = pickle.load(open('model2.pkl','rb'))

#url
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='The following candidate is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug = True)


import csv
import pickle



import numpy as np
from flask import Flask, render_template, request, g, redirect, render_template, request, session, url_for
from flask_cors import CORS,cross_origin
import pandas as pd

app=Flask(__name__)
cors=CORS(app)
learning_model = pickle.load(open('LinearRegressionModel.pkl','rb'))
data = pd.read_csv('Updated_data.csv')


@app.route('/')
def index():
    make_model= sorted(data['make/model'].unique())

    year = sorted(data['year'].unique(), reverse=True)
    condition = sorted(data['condition'].unique(), reverse=True)
    state = sorted(data['state'].apply(lambda x: x.upper()).unique())

    return render_template('index.html', make_model=make_model, year=year,condition=condition,state=state)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        make_model = request.form.get('make-model')
        year = int(request.form.get('year'))
        condition = request.form.get('condition')
        state = request.form.get('state').lower()
        miles = request.form.get('miles')

        prediction = learning_model.predict(pd.DataFrame(columns=['make/model', 'year', 'odometer', 'state', 'condition'],
                                                     data=np.array([make_model, year, miles, state, condition
                                                                    ]).reshape(1, 5)))
        if int(miles) > 299999:
            return "Your car is worth approximately $0.00"


        return "Your car is worth approximately ${:,.2f}".format(prediction[0])
    except:
        return "Please enter a value for miles"

if __name__ == "__main__":
    app.run(debug=True)

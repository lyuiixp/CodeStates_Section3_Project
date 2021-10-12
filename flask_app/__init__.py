from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('model.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/predicted', methods=['POST'])
def prediction():
    if request.method=='POST':
        n1 = request.form['num1']
        n2 = request.form['num2']
        n3 = request.form['num3']
        n4 = request.form['num4']
        n5 = request.form['num5']

        data = {'타수':[n1], '타율':[n2], '홈런':[n3], '병살':[n4], '삼진':[n5]}
        df = pd.DataFrame(data)

        path = os.getcwd() + '\\flask_app\\model\\model.pkl'
        model1 = joblib.load(path)
        #model1 = joblib.load(r'./model/model.pkl')
        predicted1 = model1.predict(df)[0].round(3)
        return render_template('model.html', predicted=predicted1)


if __name__ == "__main__":
    app.run(debug=True)

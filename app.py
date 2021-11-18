from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
#import sklearn
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib
#from scipy.sparse import hstack




app = Flask(__name__)

model = pickle.load(open('ExtraTrees_model.pkl', 'rb'))
count_vect = pickle.load(open('vectorizer.pkl', 'rb'))

sclaer = pickle.load(open('scaler.pkl', 'rb'))



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        car_name = str(request.form['car_name'])
        present_price = float(request.form['present_price'])/100000
        year = int(request.form['year'])
        Kms_Driven=np.array(int(request.form['kilometres'])).reshape(-1,1)
        owners=int(request.form['owners'])
        fuel_type=request.form['fuel_type']
        Seller_Type=request.form['Seller_Type']
        Transmission=request.form['Transmission']

        car_name=car_name.lower()
        car_name = car_name.replace(' ','_')
        car_name_vect = count_vect.transform([car_name])
        car_name_vect = car_name_vect.toarray()[0]
        print(car_name_vect)

        year=2020-year

        if(fuel_type=='Petrol'):
            fuel_type=0
        elif(fuel_type=='Diesel'):
            fuel_type=1
        else:
            fuel_type=2
        if(Seller_Type=='Individual'):
            Seller_Type=1
        else:
            Seller_Type=0
        if(Transmission=='Mannual'):
            Transmission=0
        else:
            Transmission=1
        if owners>1:
            owners=3

        Kms_Driven = sclaer.transform(Kms_Driven)
        print(present_price)
        print(fuel_type)
        print(Transmission)
        print(owners)
        print(year)
        print(Kms_Driven)

        X=[present_price,fuel_type,Seller_Type,Transmission,owners,year]
        X.extend(car_name_vect)
        X.extend(Kms_Driven[0])

        prediction=model.predict([X])
        output=round(prediction[0],2)*100000
        if output<0:
            return render_template('result.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('result.html',prediction_text="You Can Sell The Car at {} Lakh Rupees".format(round(output)))
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)



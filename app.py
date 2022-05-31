import pickle

from flask import Flask, render_template, jsonify, request
import numpy as np
model = pickle.load(open('final_model.pickle','rb'))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")



@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(2013 - float(request.form['outlet_establishment_year']))
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])
    outlet_identifier = float(request.form['outlet_identifier'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type,outlet_identifier]])
    sc = pickle.load(open('scaler.pkl', 'rb'))

    X_std = sc.transform(X)
    Y_pred = model.predict(X_std)
    final_pred = np.exp(Y_pred) -1

    return jsonify({'Prediction': float(final_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction=0)

@app.route("/predict", methods=["POST"])
def predict():

    Brand = int(request.form["Brand"])
    Year = int(request.form["Year"])
    Engine_Size = float(request.form["Engine_Size"])
    Fuel_Type = int(request.form["Fuel_Type"])
    Transmission = int(request.form["Transmission"])
    Mileage = float(request.form["Mileage"])
    Condition = int(request.form["Condition"])
    Model = int(request.form["Model"])

    features = np.array([[Brand, Year, Engine_Size,
                          Fuel_Type, Transmission,
                          Mileage, Condition, Model]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
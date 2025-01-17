from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# load the model from disk
modle_path = "model/model.pkl"
with open(modle_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # extrect data from form
    age = request.form['age']
    print(age)
    gender = request.form['gender']
    print(gender)
    stream = request.form['stream']
    print(stream)
    internships = request.form['internships']
    print(internships)
    CGPA = request.form['CGPA']
    print(CGPA)
    historyOfBacklogs = request.form['historyOfBacklogs']
    print(historyOfBacklogs)

    int_features = [int(x) for x in request.form.values()]
    print("int_features :", int_features)
    final_features = [np.array(int_features)]

    # make prediction
    prediction = model.predict(final_features)
    output = "Placed" if prediction[0] == 1 else "Not Placed"

    return render_template('index.html', age=age, gender=gender, stream=stream, internships=internships,
                           CGPA=CGPA, historyOfBacklogs=historyOfBacklogs, prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
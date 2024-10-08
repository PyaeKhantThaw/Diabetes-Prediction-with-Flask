import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Store model accuracies and models
model_accuracies = {}
models = {}

# Load the dataset from the CSV file
def load_dataset(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1]  # Features (all except the last column)
    y = data.iloc[:, -1]   # Target (last column, 'Outcome')
    return X, y

# Train the models with the uploaded dataset
def train_models(X_train, X_test, y_train, y_test):
    # SVM
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    joblib.dump(svm_model, "models/svm_model.joblib")
    model_accuracies['SVM'] = svm_accuracy
    models['SVM'] = "models/svm_model.joblib"
    
    # KNN
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    joblib.dump(knn_model, "models/knn_model.joblib")
    model_accuracies['KNN'] = knn_accuracy
    models['KNN'] = "models/knn_model.joblib"

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    joblib.dump(dt_model, "models/dt_model.joblib")
    model_accuracies['Decision Tree'] = dt_accuracy
    models['Decision Tree'] = "models/dt_model.joblib"

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    if 'dataset' not in request.files:
        return redirect(request.url)
    file = request.files['dataset']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join('data', 'dataset.csv')
        file.save(filepath)
        return redirect(url_for('loading'))

@app.route('/loading')
def loading():
    # Load dataset and split into training/testing sets
    X, y = load_dataset('data/dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    train_models(X_train, X_test, y_train, y_test)
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html', accuracies=model_accuracies)

@app.route('/predict', methods=['POST'])
def predict_page():
    selected_model = request.form.get('model')
    model = joblib.load(models[selected_model])
    
    # Render prediction input form
    return render_template('predict.html', model=selected_model)

@app.route('/predict_result', methods=['POST'])
def predict_result():
    # Get user inputs
    input_data = [float(request.form.get(attr)) for attr in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    
    # Load the selected model and predict
    selected_model = request.form.get('model')
    model = joblib.load(models[selected_model])
    prediction = model.predict([input_data])[0]
    
    # Display result
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return render_template('result_display.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import os
import pickle
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load the pickled model and feature columns
with open('philanthropy_model.pkl', 'rb') as f:
    model_bundle = pickle.load(f)
    model = model_bundle['model']
    feature_columns = model_bundle['feature_columns']

@app.route('/')
def home():
    # Update 'index.html' to use dark red/black theme
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'excel_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['excel_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_excel(file_path)

        # Preprocess input to match model's expected features
        # Fill missing engagement scores with 0
        for col in [col for col in feature_columns if 'Alumni Engagement' in col]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        # Fill missing categorical values with 'Unknown'
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].fillna('Unknown')
        if 'Circle/Crescent House' in df.columns:
            df['Circle/Crescent House'] = df['Circle/Crescent House'].fillna('Unknown')
        # Optionally group Class Year into bins (decades)
        if 'Class Year' in df.columns:
            df['Class Year Group'] = (df['Class Year'] // 10) * 10
        # One-hot encode to match training
        df_model = pd.get_dummies(df, columns=['Gender', 'Circle/Crescent House', 'Class Year Group'], drop_first=True)
        # Reindex to match model's feature columns
        X = df_model.reindex(columns=feature_columns, fill_value=0)
        # Predict
        predictions = model.predict(X)
        df['Philanthropy_Prediction'] = predictions
        output_filename = f"predictions_{file.filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df.to_excel(output_path, index=False)
        # Update 'results.html' to use dark red/black theme
        return render_template('results.html', filename=output_filename, tables=[df.head(20).to_html(classes='table table-dark table-striped', index=False)])
    return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
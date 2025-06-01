import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, send_from_directory
import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

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
        
        # Ensure all columns are included in the output
        df.to_excel(output_path, index=False)
        
        # Create a preview DataFrame with key columns for display
        preview_df = df[['Gender', 'Class Year', 'Circle/Crescent House', 'Philanthropy_Prediction']].head(20)
        # Update 'results.html' to use dark red/black theme and pass only the filename and preview table
        return render_template('results.html', 
                             filename=output_filename, 
                             tables=[preview_df.to_html(classes='table table-dark table-striped', index=False)])
    return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

@app.route('/eda', methods=['POST'])
def eda():
    if 'eda_excel_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['eda_excel_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_excel(file_path)
        sns.set(style="whitegrid")
        unique_id = str(uuid.uuid4())[:8]
        # 1. Correlation Heatmap
        engagement_cols = [
            'Alumni Engagement Score FY25',
            'Alumni Engagement Rating FY25',
            'Alumni Engagement Philanthropic FY25',
            'Alumni Engagement Volunteer FY25',
            'Alumni Engagement Experiential FY25',
            'Alumni Engagement Comms FY25'
        ]
        heatmap_img = f'engagement_correlation_heatmap_{unique_id}.png'
        try:
            corr = df[engagement_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix: Engagement Metrics & Philanthropy')
            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], heatmap_img))
            plt.close()
        except Exception:
            heatmap_img = None
        # 2. Philanthropy by Class Year
        class_year_img = f'philanthropy_by_class_year_{unique_id}.png'
        try:
            top_years = df['Class Year'].value_counts().index[:10]
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Class Year', y='Alumni Engagement Philanthropic FY25', data=df[df['Class Year'].isin(top_years)])
            plt.title('Philanthropy by Top 10 Class Years')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], class_year_img))
            plt.close()
        except Exception:
            class_year_img = None
        # 3. Philanthropy by Gender
        gender_img = f'philanthropy_by_gender_{unique_id}.png'
        try:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='Gender', y='Alumni Engagement Philanthropic FY25', data=df)
            plt.title('Philanthropy by Gender')
            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], gender_img))
            plt.close()
        except Exception:
            gender_img = None
        # 4. Philanthropy by House
        house_img = f'philanthropy_by_house_{unique_id}.png'
        try:
            top_houses = df['Circle/Crescent House'].value_counts().index[:10]
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Circle/Crescent House', y='Alumni Engagement Philanthropic FY25', data=df[df['Circle/Crescent House'].isin(top_houses)])
            plt.title('Philanthropy by Top 10 Houses')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], house_img))
            plt.close()
        except Exception:
            house_img = None
        eda_images = [img for img in [house_img, gender_img, class_year_img, heatmap_img] if img]
        return render_template('eda_results.html', eda_images=eda_images)
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/help')
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import os
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file"
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    label = predict_image(filepath)
    return render_template('index.html', prediction=label, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)

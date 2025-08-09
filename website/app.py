# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from idiyappam_untangle import process_idiyappam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXT = {'png','jpg','jpeg','bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image = None
    output_image = None
    strand_lengths = None
    if request.method == 'POST':
        file = request.files.get('image')
        pixels_per_cm = request.form.get('pixels_per_cm', None)
        if pixels_per_cm:
            try:
                pixels_per_cm = float(pixels_per_cm)
            except:
                pixels_per_cm = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            out_name, lengths_px, lengths_cm, total_len_px = process_idiyappam(upload_path, app.config['PROCESSED_FOLDER'], pixels_per_cm)
            original_image = f"uploads/{filename}"
            output_image = f"processed/{out_name}"
            if lengths_cm:
                strand_lengths = [round(x,2) for x in lengths_cm]
            else:
                strand_lengths = [round(x,2) for x in lengths_px]
    return render_template('index.html', original_image=original_image, output_image=output_image, strand_lengths=strand_lengths)

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


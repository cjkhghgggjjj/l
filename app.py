import os
import subprocess
import sys

# ===========================
# تثبيت المكتبات تلقائياً
# ===========================
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for lib in ["flask", "insightface", "onnxruntime", "opencv-python-headless", "numpy"]:
    try:
        __import__(lib.split('-')[0])
    except ImportError:
        install(lib)

# ===========================
# استدعاء المكتبات
# ===========================
from flask import Flask, request, send_file, render_template_string
import cv2
from insightface.app import FaceAnalysis

app = Flask(__name__)
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0, det_size=(640, 640))

# صفحة HTML صغيرة جدًا
HTML = """
<!DOCTYPE html>
<html>
<body style="text-align:center;font-family:sans-serif;">
<h2>كشف جنس الوجه</h2>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" required><br><br>
<button type="submit">تحليل</button>
</form>
{% if gender is not none %}
<h3>👤 الجنس: {{ 'ذكر' if gender == 1 else 'أنثى' }}</h3>
<img src="/image">
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    gender = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = "uploaded.jpg"
            file.save(path)
            img = cv2.imread(path)
            faces = face_app.get(img)
            if faces:
                gender = int(faces[0].gender)
    return render_template_string(HTML, gender=gender)

@app.route("/image")
def image():
    return send_file("uploaded.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 افتح المتصفح على: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

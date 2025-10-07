import os
import subprocess
import sys

# ===========================
# تثبيت المكتبات تلقائياً
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_libs = [
    "flask",
    "insightface",
    "onnxruntime", 
    "opencv-python-headless",
    "numpy"
]

for lib in required_libs:
    try:
        __import__(lib.split('-')[0])
    except ImportError:
        print(f"🔹 تثبيت {lib}...")
        install(lib)

# ===========================
# استدعاء المكتبات
# ===========================
from flask import Flask, render_template_string, request, send_file
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ===========================
# إعداد التطبيق
# ===========================
app = Flask(__name__)

# تحميل نموذج معرفة الجنس فقط لتقليل الحجم
face_app = FaceAnalysis(
    name='buffalo_l',  # تحديد النموذج
    providers=['CPUExecutionProvider']
)

# تهيئة النموذج مع إعدادات مبسطة
face_app.prepare(
    ctx_id=0, 
    det_size=(320, 320)  # حجم أصغر للكشف فقط
)

# ===========================
# صفحة HTML
# ===========================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
  <meta charset="UTF-8">
  <title>تحليل الجنس - InsightFace</title>
  <style>
    body {font-family: Arial; text-align:center; background:#f5f5f5;}
    h2 {color:#333;}
    form {margin:30px auto; padding:20px; background:white; border-radius:15px; width:350px; box-shadow:0 0 10px #ccc;}
    input[type=file]{margin:10px;}
    img {margin-top:20px; width:250px; border-radius:10px;}
    .info {background:#fff; display:inline-block; margin-top:20px; padding:15px; border-radius:10px; box-shadow:0 0 5px #aaa;}
    .male {color: blue; font-weight: bold;}
    .female {color: pink; font-weight: bold;}
  </style>
</head>
<body>
  <h2>تحليل الجنس باستخدام InsightFace</h2>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br><br>
    <button type="submit">تحليل الجنس</button>
  </form>
  {% if result %}
    <div class="info">
      <h3>👤 النتيجة:</h3>
      <p class="{{ 'male' if result.gender == 1 else 'female' }}">
        الجنس: {{ 'ذكر' if result.gender == 1 else 'أنثى' }}
      </p>
      <p>عدد الوجوه المكتشفة: {{ result.faces }}</p>
      <img src="{{ image_url }}">
    </div>
  {% endif %}
</body>
</html>
"""

# ===========================
# مسارات التطبيق
# ===========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = "uploaded.jpg"
            file.save(path)

            img = cv2.imread(path)
            faces = face_app.get(img)

            if len(faces) == 0:
                return render_template_string(HTML_PAGE, result=None, image_url=None)

            face = faces[0]
            result = type("Result", (), {})()
            
            # الحصول على الجنس
            result.gender = int(face.gender)
            result.faces = len(faces)

            return render_template_string(HTML_PAGE, result=result, image_url="/image")
    
    return render_template_string(HTML_PAGE, result=None, image_url=None)

@app.route("/image")
def serve_image():
    return send_file("uploaded.jpg", mimetype="image/jpeg")

# ===========================
# تشغيل التطبيق
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 افتح المتصفح على: http://0.0.0.0:{port}")
    print(f"🔍 النموذج مضبوط لتحليل الجنس")
    app.run(host="0.0.0.0", port=port, debug=False)

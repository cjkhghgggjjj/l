import os
import subprocess
import sys

# ===========================
# 🔹 تثبيت المكتبات تلقائيًا
# ===========================
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"⚠️ فشل تثبيت {package}: {e}")

required_libs = [
    "flask",
    "insightface",
    "onnxruntime",
    "opencv-python-headless",  # لتجنب مشاكل واجهات OpenCV الرسومية
    "numpy"
]

for lib in required_libs:
    try:
        __import__(lib.split('-')[0])
    except ImportError:
        print(f"🔹 تثبيت {lib}...")
        install(lib)

# ===========================
# 📦 استدعاء المكتبات
# ===========================
from flask import Flask, render_template_string, request, send_file
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ===========================
# ⚙️ إعداد التطبيق
# ===========================
app = Flask(__name__)

# ✅ تحميل نموذج خفيف وصغير لتقليل الذاكرة
print("🧠 تحميل نموذج InsightFace الصغير (buffalo_s)...")
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU فقط + دقة منخفضة لتقليل الذاكرة
print("✅ تم تحميل النموذج بنجاح.")

# ===========================
# 🌐 واجهة HTML بسيطة
# ===========================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
  <meta charset="UTF-8">
  <title>تحليل ملامح الوجه - InsightFace</title>
  <style>
    body {font-family: Arial; text-align:center; background:#f0f0f0; margin-top:40px;}
    h2 {color:#333;}
    form {margin:30px auto; padding:20px; background:white; border-radius:15px; width:350px; box-shadow:0 0 10px #ccc;}
    input[type=file]{margin:10px;}
    button {padding:10px 20px; border:none; background:#3498db; color:white; border-radius:10px; cursor:pointer;}
    button:hover {background:#2980b9;}
    img {margin-top:20px; width:250px; border-radius:10px;}
    .info {background:#fff; display:inline-block; margin-top:20px; padding:15px; border-radius:10px; box-shadow:0 0 5px #aaa;}
  </style>
</head>
<body>
  <h2>تحليل ملامح الوجه باستخدام InsightFace</h2>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br><br>
    <button type="submit">تحليل الصورة</button>
  </form>
  {% if result %}
    <div class="info">
      <h3>👤 النتيجة:</h3>
      <p>العمر التقريبي: {{ result.age }}</p>
      <p>الجنس: {{ 'ذكر' if result.gender == 1 else 'أنثى' }}</p>
      <p>عدد الوجوه المكتشفة: {{ result.faces }}</p>
      <img src="{{ image_url }}">
    </div>
  {% elif noface %}
    <div class="info">
      <p>❌ لم يتم اكتشاف أي وجه في الصورة.</p>
    </div>
  {% endif %}
</body>
</html>
"""

# ===========================
# 🧩 المسارات
# ===========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = "uploaded.jpg"
            file.save(path)

            # ✅ تقليل حجم الصورة لتقليل الذاكرة
            img = cv2.imread(path)
            if img is None:
                return render_template_string(HTML_PAGE, result=None, noface=True)

            img = cv2.resize(img, (320, 320))  # تصغير الصورة
            faces = face_app.get(img)

            if len(faces) == 0:
                print("⚠️ لم يتم اكتشاف أي وجه في الصورة.")
                return render_template_string(HTML_PAGE, result=None, noface=True)

            face = faces[0]
            result = type("Result", (), {})()
            result.age = int(face.age)
            result.gender = int(face.gender)
            result.faces = len(faces)

            print(f"✅ تم اكتشاف {len(faces)} وجه(وجوه) - العمر: {result.age}, الجنس: {'ذكر' if result.gender == 1 else 'أنثى'}")

            return render_template_string(HTML_PAGE, result=result, image_url="/image", noface=False)
    return render_template_string(HTML_PAGE, result=None, noface=False)

@app.route("/image")
def serve_image():
    return send_file("uploaded.jpg", mimetype="image/jpeg")

# ===========================
# 🚀 تشغيل التطبيق
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("===================================")
    print("🚀 تطبيق تحليل الوجه يعمل الآن!")
    print(f"🌍 افتح المتصفح على: http://0.0.0.0:{port}")
    print("===================================")
    app.run(host="0.0.0.0", port=port)

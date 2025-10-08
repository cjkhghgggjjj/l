import os
import subprocess
import sys
import requests
import base64
from flask import Flask, request, render_template_string
import cv2
import numpy as np

# ===========================
# تثبيت المكتبات تلقائيًا
# ===========================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import insightface
except:
    install("insightface")
    import insightface

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# روابط ملفات النماذج
# ===========================
model_files = {
    "scrfd": "https://classy-douhua-0d9950.netlify.app/scrfd_10g_bnkps.onnx.index.js",
    "glintr100": "https://classy-douhua-0d9950.netlify.app/glintr100.onnx.index.js",
    "genderage": "https://classy-douhua-0d9950.netlify.app/genderage.onnx.index.js",
    "2d106det": "https://classy-douhua-0d9950.netlify.app/2d106det.onnx.index.js",
    "1k3d68": "https://classy-douhua-0d9950.netlify.app/1k3d68.onnx.index.js"
}

tmp_model_dir = "/tmp/insightface_models"
os.makedirs(tmp_model_dir, exist_ok=True)

# ===========================
# تحميل وفك ملفات ONNX من الروابط
# ===========================
def download_model_from_js(name, url):
    tmp_path = os.path.join(tmp_model_dir, f"{name}.onnx")
    if os.path.exists(tmp_path):
        return tmp_path  # إذا موجود بالفعل
    print(f"📥 تنزيل نموذج {name} من الرابط...")
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"فشل تنزيل النموذج {name}")
    content = r.text

    # استخراج Base64 من ملف index.js
    # افترض أن الملف يحتوي على: const MODEL = "BASE64_STRING";
    start = content.find('"') + 1
    end = content.rfind('"')
    base64_data = content[start:end]

    with open(tmp_path, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return tmp_path

# تحميل جميع النماذج المطلوبة
for name, url in model_files.items():
    download_model_from_js(name, url)

# ===========================
# تهيئة نموذج FaceAnalysis
# ===========================
model = insightface.app.FaceAnalysis(name="antelopev2")
model.prepare(ctx_id=-1)  # CPU فقط

# ===========================
# HTML صفحة الرفع
# ===========================
HTML_PAGE = """
<!doctype html>
<title>كشف جنس الوجه</title>
<h2>رفع صورة لتحديد الجنس</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=رفع>
</form>
{% if gender %}
<h3>النتيجة: {{ gender }}</h3>
<img src="{{ image_url }}" width="300">
{% endif %}
"""

# ===========================
# رفع الصورة وتحليل الجنس
# ===========================
@app.route("/", methods=["GET", "POST"])
def index():
    gender_result = None
    image_url = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_url = filepath

            img = cv2.imread(filepath)
            faces = model.get(img)

            if len(faces) == 0:
                gender_result = "🚫 لم يتم اكتشاف أي وجه"
            else:
                face = faces[0]
                gender_result = "ذكر" if face.gender == 1 else "أنثى"

    return render_template_string(HTML_PAGE, gender=gender_result, image_url=image_url)

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

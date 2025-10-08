import os
import subprocess
import sys
from flask import Flask, request, render_template_string
import cv2

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

try:
    import numpy as np
except:
    install("numpy")
    import numpy as np

try:
    from flask import Flask, request, render_template_string
except:
    install("flask")
    from flask import Flask, request, render_template_string

# ===========================
# إعداد Flask
# ===========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# تحميل نموذج خفيف لكشف الجنس فقط
# ===========================
# استخدام نموذج RetinaFace خفيف جدًا مع FaceAnalysis
model = insightface.app.FaceAnalysis(name="retinaface_mnet025_v2")
model.prepare(ctx_id=-1)  # CPU فقط

# صفحة HTML بسيطة
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
# صفحة الرفع والتحليل
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

            # قراءة الصورة
            img = cv2.imread(filepath)

            # كشف الوجه
            faces = model.get(img)
            if len(faces) == 0:
                gender_result = "🚫 لم يتم اكتشاف أي وجه"
            else:
                # أول وجه فقط
                face = faces[0]
                gender_result = "ذكر" if face.gender == 1 else "أنثى"

    return render_template_string(HTML_PAGE, gender=gender_result, image_url=image_url)

# ===========================
# تشغيل الخادم
# ===========================
if __name__ == "__main__":
    # Render قد يتطلب استخدام PORT من البيئة
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

import os
import subprocess
import sys
import traceback

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
from flask import Flask, render_template_string, request, send_file, jsonify
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ===========================
# إعداد التطبيق
# ===========================
app = Flask(__name__)

# تهيئة النموذج مع معالجة الأخطاء
try:
    face_app = FaceAnalysis(
        name='buffalo_l',  # استخدام buffalo_l لأنه أكثر استقراراً
        providers=['CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    print("✅ النموذج جاهز للاستخدام")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
    face_app = None

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
    .error {background:#ffe6e6; color:#d00; padding:15px; border-radius:10px;}
    .male {color: blue; font-weight: bold;}
    .female {color: pink; font-weight: bold;}
  </style>
</head>
<body>
  <h2>تحليل الجنس باستخدام InsightFace</h2>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br><br>
    <button type="submit">تحليل الصورة</button>
  </form>
  
  {% if error %}
    <div class="error">
      <h3>⚠️ خطأ:</h3>
      <p>{{ error }}</p>
    </div>
  {% endif %}
  
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
    try:
        if request.method == "POST":
            # التحقق من وجود النموذج
            if face_app is None:
                return render_template_string(HTML_PAGE, error="النموذج غير جاهز. يرجى المحاولة لاحقاً.")
            
            file = request.files["image"]
            if file:
                # حفظ الصورة
                path = "uploaded.jpg"
                file.save(path)
                
                # قراءة الصورة
                img = cv2.imread(path)
                if img is None:
                    return render_template_string(HTML_PAGE, error="تعذر قراءة الصورة. يرجى تحميل صورة صالحة.")
                
                # تحليل الوجه
                faces = face_app.get(img)
                
                if len(faces) == 0:
                    return render_template_string(HTML_PAGE, error="لم يتم العثور على أي وجه في الصورة.")
                
                # الحصول على النتائج
                face = faces[0]
                result = {
                    'gender': int(face.gender),
                    'faces': len(faces)
                }
                
                return render_template_string(HTML_PAGE, result=result, image_url="/image")
        
        return render_template_string(HTML_PAGE, result=None, image_url=None, error=None)
    
    except Exception as e:
        print(f"❌ خطأ في المعالجة: {e}")
        print(traceback.format_exc())
        return render_template_string(HTML_PAGE, error=f"حدث خطأ في المعالجة: {str(e)}")

@app.route("/image")
def serve_image():
    try:
        return send_file("uploaded.jpg", mimetype="image/jpeg")
    except:
        return "الصورة غير متوفرة", 404

@app.route("/health")
def health_check():
    """فحص حالة التطبيق"""
    status = {
        "model_loaded": face_app is not None,
        "status": "ready" if face_app else "error"
    }
    return jsonify(status)

# ===========================
# تشغيل التطبيق
# ===========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 افتح المتصفح على: http://0.0.0.0:{port}")
    print(f"🔍 حالة النموذج: {'✅ جاهز' if face_app else '❌ خطأ'}")
    
    if face_app is None:
        print("❌ لم يتم تحميل النموذج. تأكد من:")
        print("   - اتصال الإنترنت لتحميل النماذج")
        print("   - مساحة تخزين كافية")
        print("   - صلاحيات الكتابة في المجلد")
    
    app.run(host="0.0.0.0", port=port, debug=False)

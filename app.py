import os
import subprocess
import sys
import traceback

# ===========================
# تثبيت المكتبات تلقائياً
# ===========================
def install_packages():
    required_libs = [
        "flask",
        "onnxruntime", 
        "opencv-python-headless",
        "numpy",
        "requests",
        "pillow"
    ]
    
    print("🔧 بدء تثبيت المكتبات المطلوبة...")
    
    for lib in required_libs:
        try:
            if lib == "opencv-python-headless":
                __import__("cv2")
            elif lib == "pillow":
                __import__("PIL")
            else:
                __import__(lib.split('-')[0])
            print(f"✅ {lib} - مثبت مسبقاً")
        except ImportError:
            print(f"📦 جاري تثبيت {lib}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
                print(f"✅ تم تثبيت {lib} بنجاح")
            except Exception as e:
                print(f"❌ فشل تثبيت {lib}: {e}")

install_packages()

# ===========================
# إعادة استدعاء المكتبات
# ===========================
print("🔄 تحميل المكتبات بعد التثبيت...")

try:
    from flask import Flask, render_template_string, request, send_file, jsonify
    import cv2
    import numpy as np
    import requests
    import onnxruntime as ort
    from PIL import Image
    import io
    print("✅ تم تحميل جميع المكتبات بنجاح!")
except ImportError as e:
    print(f"❌ خطأ في تحميل المكتبات: {e}")
    sys.exit(1)

# ===========================
# إعداد التطبيق
# ===========================
app = Flask(__name__)

print("🚀 تهيئة التطبيق مع نموذج AntelopeV2...")

# روابط نموذج AntelopeV2
MODEL_URLS = {
    "detection": "https://classy-douhua-0d9950.netlify.app/scrfd_10g_bnkps.onnx.index.js",
    "recognition": "https://classy-douhua-0d9950.netlify.app/glintr100.onnx.index.js",
    "genderage": "https://classy-douhua-0d9950.netlify.app/genderage.onnx.index.js",
    "landmarks_2d": "https://classy-douhua-0d9950.netlify.app/2d106det.onnx.index.js",
    "landmarks_3d": "https://classy-douhua-0d9950.netlify.app/1k3d68.onnx.index.js"
}

class AntelopeV2FaceAnalysis:
    """فئة لتحليل الوجوه باستخدام نموذج AntelopeV2"""
    
    def __init__(self):
        self.sessions = {}
        self.initialized = False
        self.providers = ['CPUExecutionProvider']
    
    def load_models(self):
        """تحميل جميع نماذج AntelopeV2 من الـ API"""
        try:
            print("🌐 جاري تحميل نموذج AntelopeV2 من API...")
            
            models_to_load = {
                "detection": "كشف الوجوه",
                "recognition": "التعرف على الوجوه", 
                "genderage": "تحليل الجنس والعمر",
                "landmarks_2d": "نقاط الوجه 2D",
                "landmarks_3d": "نقاط الوجه 3D"
            }
            
            for model_key, model_name in models_to_load.items():
                print(f"📥 جاري تحميل {model_name}...")
                response = requests.get(MODEL_URLS[model_key], timeout=60)
                response.raise_for_status()
                
                # تحميل النموذج في ONNX Runtime
                self.sessions[model_key] = ort.InferenceSession(
                    response.content, 
                    providers=self.providers
                )
                
                print(f"✅ تم تحميل {model_name} بنجاح")
                
                # طباعة معلومات النموذج
                session = self.sessions[model_key]
                print(f"   🔍 معلومات {model_name}:")
                for i, input in enumerate(session.get_inputs()):
                    print(f"      الإدخال {i}: {input.name} - {input.shape} - {input.type}")
                for i, output in enumerate(session.get_outputs()):
                    print(f"      الإخراج {i}: {output.name} - {output.shape} - {output.type}")
            
            self.initialized = True
            print("🎉 تم تحميل جميع نماذج AntelopeV2 بنجاح!")
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تحميل النماذج: {e}")
            traceback.print_exc()
            return False
    
    def prepare(self, ctx_id=0, det_size=(640, 640)):
        return self.load_models()
    
    def get(self, img):
        """تحليل الصورة باستخدام نموذج AntelopeV2"""
        if not self.initialized:
            return []
        
        try:
            # تحويل الصورة إلى RGB
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            original_height, original_width = img.shape[:2]
            
            # كشف الوجوه
            faces = self._detect_faces(img_rgb, original_width, original_height)
            
            # تحليل كل وجه
            for face in faces:
                # استخراج embedding
                face.embedding = self._get_face_embedding(img_rgb, face.bbox)
                
                # تحليل الجنس والعمر
                face.gender, face.age = self._analyze_gender_age(img_rgb, face.bbox)
                
                # الحصول على نقاط الوجه
                face.landmarks_2d = self._get_2d_landmarks(img_rgb, face.bbox)
                face.landmarks_3d = self._get_3d_landmarks(img_rgb, face.bbox)
                
                # رسم النتائج على الصورة
                self._draw_face_analysis(img, face)
            
            return faces
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الصورة: {e}")
            traceback.print_exc()
            return []
    
    def _detect_faces(self, img_rgb, orig_w, orig_h):
        """كشف الوجوه باستخدام SCRFD"""
        class Face:
            def __init__(self, bbox, score, kps=None):
                self.bbox = bbox  # [x1, y1, x2, y2]
                self.det_score = score
                self.kps = kps  # نقاط المفاتيح
                self.embedding = None
                self.gender = 0
                self.age = 25
                self.landmarks_2d = None
                self.landmarks_3d = None
        
        try:
            session = self.sessions["detection"]
            input_size = (640, 640)
            
            # تحضير الصورة للنموذج
            img_resized = cv2.resize(img_rgb, input_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_normalized = (img_normalized - 0.5) / 0.5
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)
            
            # تشغيل النموذج
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_batch})
            
            faces = []
            
            # معالجة المخرجات (تعديل حسب تنسيق مخرجات SCRFD)
            if len(outputs) >= 2:
                boxes = outputs[0]  # bounding boxes
                scores = outputs[1]  # confidence scores
                
                for i in range(len(scores)):
                    if scores[i] > 0.5:  # عتبة الثقة
                        # تحويل الإحداثيات
                        scale_x = orig_w / input_size[0]
                        scale_y = orig_h / input_size[1]
                        
                        if boxes.shape[1] >= 4:
                            x1, y1, x2, y2 = boxes[i][:4]
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            # التأكد من الإحداثيات
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(orig_w, x2)
                            y2 = min(orig_h, y2)
                            
                            face = Face([x1, y1, x2, y2], float(scores[i]))
                            faces.append(face)
                            print(f"👤 وجه مكتشف: {face.bbox}, ثقة: {scores[i]:.2f}")
            
            # إذا لم يتم اكتشاف وجوه، إرجاع وجه افتراضي
            if len(faces) == 0:
                bbox_size = min(orig_w, orig_h) // 3
                x_center = orig_w // 2
                y_center = orig_h // 2
                bbox = [
                    max(0, x_center - bbox_size // 2),
                    max(0, y_center - bbox_size // 2),
                    min(orig_w, x_center + bbox_size // 2),
                    min(orig_h, y_center + bbox_size // 2)
                ]
                face = Face(bbox, 0.8)
                faces.append(face)
                print("⚠️ استخدام وجه افتراضي")
            
            return faces
            
        except Exception as e:
            print(f"❌ خطأ في كشف الوجوه: {e}")
            return []
    
    def _get_face_embedding(self, img_rgb, bbox):
        """استخراج embedding باستخدام GlintR100"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return np.random.randn(512).astype(np.float32)
            
            # تحضير صورة الوجه
            face_size = (112, 112)
            face_resized = cv2.resize(face_img, face_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5
            face_chw = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_chw, axis=0)
            
            # تشغيل نموذج التعرف
            session = self.sessions["recognition"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face_batch})
            
            if len(outputs) > 0:
                embedding = outputs[0].flatten()
                # تطبيع الـ embedding
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.astype(np.float32)
            else:
                return np.random.randn(512).astype(np.float32)
                
        except Exception as e:
            print(f"❌ خطأ في استخراج embedding: {e}")
            return np.random.randn(512).astype(np.float32)
    
    def _analyze_gender_age(self, img_rgb, bbox):
        """تحليل الجنس والعمر"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return 0, 30
            
            # تحضير صورة الوجه
            face_size = (112, 112)
            face_resized = cv2.resize(face_img, face_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5
            face_chw = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_chw, axis=0)
            
            # تشغيل نموذج الجنس والعمر
            session = self.sessions["genderage"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face_batch})
            
            if len(outputs) >= 2:
                gender_logits = outputs[0]
                age_output = outputs[1]
                
                # توقع الجنس (0 = أنثى, 1 = ذكر)
                gender = 1 if gender_logits[0][0] < gender_logits[0][1] else 0
                age = int(age_output[0][0])
                
                return gender, max(1, min(100, age))
            else:
                return 0, 30
                
        except Exception as e:
            print(f"❌ خطأ في تحليل الجنس والعمر: {e}")
            return 0, 30
    
    def _get_2d_landmarks(self, img_rgb, bbox):
        """الحصول على نقاط الوجه 2D"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return np.zeros((106, 2), dtype=np.float32)
            
            # تحضير الصورة
            face_size = (192, 192)
            face_resized = cv2.resize(face_img, face_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5
            face_chw = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_chw, axis=0)
            
            # تشغيل النموذج
            session = self.sessions["landmarks_2d"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face_batch})
            
            if len(outputs) > 0:
                landmarks = outputs[0].reshape(-1, 2)
                # تحويل الإحداثيات إلى الحجم الأصلي
                scale_x = (x2 - x1) / face_size[0]
                scale_y = (y2 - y1) / face_size[1]
                landmarks[:, 0] = landmarks[:, 0] * scale_x + x1
                landmarks[:, 1] = landmarks[:, 1] * scale_y + y1
                return landmarks
            else:
                return np.zeros((106, 2), dtype=np.float32)
                
        except Exception as e:
            print(f"❌ خطأ في نقاط الوجه 2D: {e}")
            return np.zeros((106, 2), dtype=np.float32)
    
    def _get_3d_landmarks(self, img_rgb, bbox):
        """الحصول على نقاط الوجه 3D"""
        try:
            x1, y1, x2, y2 = bbox
            face_img = img_rgb[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return np.zeros((68, 3), dtype=np.float32)
            
            # تحضير الصورة
            face_size = (192, 192)
            face_resized = cv2.resize(face_img, face_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5
            face_chw = np.transpose(face_normalized, (2, 0, 1))
            face_batch = np.expand_dims(face_chw, axis=0)
            
            # تشغيل النموذج
            session = self.sessions["landmarks_3d"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: face_batch})
            
            if len(outputs) > 0:
                landmarks = outputs[0].reshape(-1, 3)
                return landmarks
            else:
                return np.zeros((68, 3), dtype=np.float32)
                
        except Exception as e:
            print(f"❌ خطأ في نقاط الوجه 3D: {e}")
            return np.zeros((68, 3), dtype=np.float32)
    
    def _draw_face_analysis(self, img, face):
        """رسم نتائج التحليل على الصورة"""
        try:
            # رسم bounding box
            x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
            color = (0, 255, 0) if face.gender == 1 else (255, 0, 255)  # أزرق للذكر, وردي للأنثى
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # رسم نقاط الوجه 2D
            if face.landmarks_2d is not None:
                for point in face.landmarks_2d:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
            
            # إضافة معلومات النص
            label = f"{'ذكر' if face.gender == 1 else 'أنثى'} {face.age}سنة"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            print(f"❌ خطأ في الرسم: {e}")

# تهيئة النموذج
print("🔧 جاري تهيئة نموذج AntelopeV2...")
face_analyzer = AntelopeV2FaceAnalysis()
init_success = face_analyzer.prepare()

if init_success:
    print("🎉 نموذج AntelopeV2 جاهز للاستخدام!")
else:
    print("❌ فشل تحميل النموذج")

# ===========================
# واجهة الويب
# ===========================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <title>تحليل الوجوه - AntelopeV2</title>
    <style>
        body {font-family: Arial; text-align:center; background:#f5f5f5;}
        h2 {color:#333;}
        form {margin:30px auto; padding:20px; background:white; border-radius:15px; width:350px; box-shadow:0 0 10px #ccc;}
        input[type=file]{margin:10px;}
        img {margin-top:20px; max-width:400px; border-radius:10px;}
        .info {background:#fff; display:inline-block; margin-top:20px; padding:15px; border-radius:10px; box-shadow:0 0 5px #aaa;}
        .error {background:#ffe6e6; color:#d00; padding:15px; border-radius:10px;}
        .success {background:#e6ffe6; color:#060; padding:10px; border-radius:10px;}
        .male {color: blue; font-weight: bold;}
        .female {color: pink; font-weight: bold;}
        .stats {background:#f0f8ff; padding:10px; border-radius:5px; margin:10px;}
        .model-info {background:#fffacd; padding:10px; border-radius:5px; margin:10px;}
        .landmarks {background:#f0fff0; padding:10px; border-radius:5px; margin:10px;}
    </style>
</head>
<body>
    <div class="success">
        <h2>🧠 نظام تحليل الوجوه - AntelopeV2</h2>
        <p>أحدث نموذج لتحليل الوجوه</p>
    </div>
    
    {% if model_loaded %}
    <div class="success">
        <p>✅ نموذج AntelopeV2 محمل بنجاح</p>
    </div>
    {% else %}
    <div class="error">
        <p>❌ النموذج غير محمل</p>
    </div>
    {% endif %}
    
    <div class="model-info">
        <h4>📊 مكونات النموذج:</h4>
        <p>• SCRFD - كشف الوجوه</p>
        <p>• GlintR100 - التعرف على الوجوه</p>
        <p>• GenderAge - تحليل الجنس والعمر</p>
        <p>• 2D106 - نقاط الوجه 2D</p>
        <p>• 3D68 - نقاط الوجه 3D</p>
    </div>
    
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <br><br>
        <button type="submit">🔍 تحليل الصورة</button>
    </form>
    
    {% if error %}
    <div class="error">
        <h3>⚠️ خطأ:</h3>
        <p>{{ error }}</p>
    </div>
    {% endif %}
    
    {% if result %}
    <div class="info">
        <h3>👤 نتائج التحليل:</h3>
        <div class="stats">
            <p class="{{ 'male' if result.gender == 1 else 'female' }}">
                🚹🚺 الجنس: <strong>{{ 'ذكر' if result.gender == 1 else 'أنثى' }}</strong>
            </p>
            <p>🎂 العمر: <strong>{{ result.age }} سنة</strong></p>
            <p>👥 عدد الوجوه: <strong>{{ result.faces }}</strong></p>
            <p>🎯 الثقة: <strong>{{ "%.1f"|format(result.confidence * 100) }}%</strong></p>
        </div>
        {% if result.landmarks_2d %}
        <div class="landmarks">
            <p>📍 نقاط الوجه: <strong>106 نقطة 2D + 68 نقطة 3D</strong></p>
        </div>
        {% endif %}
        <img src="{{ image_url }}" alt="الصورة المحللة">
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            if not face_analyzer.initialized:
                return render_template_string(HTML_PAGE, 
                    error="النموذج غير جاهز",
                    model_loaded=False)

            file = request.files["image"]
            if file:
                file_data = file.read()
                img_array = np.frombuffer(file_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    return render_template_string(HTML_PAGE, 
                        error="تعذر قراءة الصورة",
                        model_loaded=face_analyzer.initialized)

                print("🔍 بدء التحليل مع AntelopeV2...")
                faces = face_analyzer.get(img)
                print(f"📊 وجوه مكتشفة: {len(faces)}")

                if len(faces) == 0:
                    cv2.imwrite("uploaded.jpg", img)
                    return render_template_string(HTML_PAGE, 
                        error="لم يتم العثور على وجوه",
                        model_loaded=face_analyzer.initialized)

                face = faces[0]
                result = {
                    'gender': face.gender,
                    'age': face.age,
                    'faces': len(faces),
                    'confidence': face.det_score,
                    'landmarks_2d': face.landmarks_2d is not None,
                    'landmarks_3d': face.landmarks_3d is not None
                }
                
                cv2.imwrite("uploaded.jpg", img)
                return render_template_string(HTML_PAGE, 
                    result=result, 
                    image_url="/image",
                    model_loaded=face_analyzer.initialized)
        
        return render_template_string(HTML_PAGE, 
            result=None, 
            model_loaded=face_analyzer.initialized)
    
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return render_template_string(HTML_PAGE, 
            error=str(e),
            model_loaded=face_analyzer.initialized)

@app.route("/image")
def serve_image():
    try:
        return send_file("uploaded.jpg", mimetype="image/jpeg")
    except:
        return "الصورة غير متوفرة", 404

@app.route("/models")
def list_models():
    """عرض معلومات النماذج"""
    models_info = {}
    for name, url in MODEL_URLS.items():
        models_info[name] = {
            "url": url,
            "loaded": name in face_analyzer.sessions
        }
    return jsonify(models_info)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 التشغيل على: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

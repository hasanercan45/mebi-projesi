from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- YENİ MODEL: OpenCV'nin LBPH Yüz Tanıyıcısı ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Etiketleri ve isimleri saklamak için
label_map = {}
current_id = 0

print("Yüzler öğreniliyor...")
known_faces_dir = "static/known_faces"

faces, ids = [], []

if os.path.exists(known_faces_dir):
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name not in label_map.values():
            label_map[current_id] = person_name
            person_id = current_id
            current_id += 1

        for filename in os.listdir(person_dir):
            if not (filename.endswith((".jpg", ".jpeg", ".png"))):
                continue

            image_path = os.path.join(person_dir, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            detected_faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                faces.append(img[y:y+h, x:x+w])
                ids.append(person_id)

if len(faces) > 0:
    print(f"{len(np.unique(ids))} kişiye ait {len(faces)} yüz örneği ile model eğitiliyor...")
    recognizer.train(faces, np.array(ids))
    print("Model başarıyla eğitildi.")
else:
    print("UYARI: Eğitim için hiç yüz bulunamadı.")

print("Sunucu başlatılıyor.")

@app.route('/yoklama_yap', methods=['POST'])
def yoklama_yap():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]

        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        response_data = { "name": "Bilinmeyen Kişi" }

        for (x, y, w, h) in detected_faces:
            # Tanıma yap
            id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])

            # Güven skoru ne kadar düşükse, eşleşme o kadar iyidir.
            if confidence < 70: # Bu eşik değeriyle oynayabilirsiniz (örn: 50 daha hassas)
                name = label_map.get(id, "Bilinmeyen Kişi")
                response_data["name"] = name.capitalize()
                response_data["status"] = f"Hoş geldin, {name.capitalize()}!"
            else:
                response_data["name"] = "Bilinmeyen Kişi"
                response_data["status"] = "Yüz algılandı, ancak kayıtlı değil."

            # Sadece ilk bulunan yüzü dikkate al
            break 

        return jsonify(response_data)

    except Exception as e:
        print(f"Sunucu hatası: {e}")
        return jsonify({'error': 'Sunucuda bir hata oluştu.', 'details': str(e)}), 500
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
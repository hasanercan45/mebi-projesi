from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
from deepface import DeepFace
import shutil

app = Flask(__name__)
CORS(app) # Tarayıcıdan gelen isteklere izin vermek için

# DeepFace'in model dosyalarını indirebileceği ve kullanabileceği geçici bir alan
# Render gibi platformlarda /tmp/ klasörü yazılabilirdir.
HOME = '/tmp/'
os.environ['DEEPFACE_HOME'] = HOME

# --- YÜZLERİ ÖĞRENME AŞAMASI (Sadece sunucu ilk başladığında çalışır) ---
known_face_encodings = []
known_face_names = []
print("Yüzler yükleniyor...")
known_faces_dir = "static/known_faces"

if os.path.exists(known_faces_dir):
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not (filename.endswith((".jpg", ".jpeg", ".png"))):
                continue

            image_path = os.path.join(person_dir, filename)
            try:
                # DeepFace yüzleri doğrudan yoldan tanıyabildiği için ön yüklemeye gerek yok.
                # Bu listeyi gelecekteki olası optimizasyonlar için boş bırakıyoruz.
                # Asıl işlem /yoklama_yap içinde gerçekleşecek.
                print(f"-> {person_name} için yüz dosyası bulundu: {filename}")
            except Exception as e:
                print(f"Hata: {image_path} dosyasıyla ilgili bir sorun var. Hata: {e}")
else:
    print(f"UYARI: '{known_faces_dir}' klasörü bulunamadı.")

print("Yüz veritabanı hazır. Sunucu başlatılıyor.")
# --- ÖĞRENME AŞAMASI BİTTİ ---

@app.route('/')
def ana_sayfa():
    return "Mebi Yüz Tanıma Sunucusu Aktif!"

@app.route('/yoklama_yap', methods=['POST'])
def yoklama_yap():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Resim verisi bulunamadı.'}), 400

        image_data = data['image'].split(',')[1]

        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        db_path = "static/known_faces"

        dfs = DeepFace.find(
            img_path=img, 
            db_path=db_path, 
            model_name='VGG-Face', 
            enforce_detection=False,
            silent=True # Konsolu temiz tutmak için
        )

        response_data = {
            "status": "Yüz algılandı, ancak kayıtlı değil.",
            "name": "Bilinmeyen Kişi",
            "confidence": None
        }

        if dfs and not dfs[0].empty:
            best_match = dfs[0].iloc[0]
            identity = best_match['identity']
            name = os.path.basename(os.path.dirname(identity))

            response_data["status"] = f"Hoş geldin, {name.capitalize()}!"
            response_data["name"] = name.capitalize()

        # Geçici dosyaları temizle
        if os.path.exists(os.path.join(HOME, ".deepface")):
            shutil.rmtree(os.path.join(HOME, ".deepface"))

        return jsonify(response_data)

    except Exception as e:
        print(f"Sunucu hatası: {e}")
        return jsonify({'error': 'Sunucuda bir hata oluştu.', 'details': str(e)}), 500

# Bu kısım Render'ın sunucuyu nasıl çalıştıracağını belirler
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
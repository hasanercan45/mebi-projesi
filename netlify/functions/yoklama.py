import json
import base64
import cv2
import numpy as np
import face_recognition
import os

# --- YÜZLERİ ÖĞRENME AŞAMASI ---
# Bu kısım fonksiyon her çağrıldığında yeniden çalışacak.
# Daha performanslı yöntemler olsa da, projemiz için bu yeterlidir.
known_face_encodings = []
known_face_names = []

# Klasör yolu, fonksiyonun çalıştığı yere göre ayarlanır.
# Netlify, projenin ana dizinini baz alır.
known_faces_dir = "static/known_faces"

# known_faces klasöründeki her bir alt klasör (kişi) için döngü başlat
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    # O kişiye ait klasördeki her bir fotoğraf için döngü başlat
    for filename in os.listdir(person_dir):
        if not (filename.endswith((".jpg", ".jpeg", ".png"))):
            continue
            
        image_path = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
        except Exception as e:
            print(f"Hata: {image_path} dosyası yüklenemedi veya işlenemedi. Hata: {e}")

print("Bilinen yüzler yüklendi.")
# --- ÖĞRENME AŞAMASI BİTTİ ---

# --- NETLIFY HANDLER FONKSİYONU ---
def handler(event, context):
    try:
        # Gelen isteğin gövdesini (body) al ve JSON olarak ayrıştır
        body = json.loads(event['body'])
        image_data = body['image'].split(',')[1]

        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
        
        # Varsayılan cevap
        response_data = {
            "status": "Yüz Algılanamadı. Lütfen Kameraya Bakın.",
            "name": None,
            "confidence": None
        }
        
        if len(face_locations) > 1:
            response_data["status"] = "Birden Fazla Yüz Algılandı. Lütfen Tek Kişi Olun."
        elif len(face_locations) == 1 and unknown_face_encodings:
            unknown_encoding = unknown_face_encodings[0]
            
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
            
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            
            TOLERANCE = 0.6
            
            if best_distance <= TOLERANCE:
                name = known_face_names[best_match_index]
                confidence = round((1 - best_distance) * 100)
                
                response_data["status"] = f"Hoş geldin, {name.capitalize()}!"
                response_data["name"] = name.capitalize()
                response_data["confidence"] = confidence
            else:
                response_data["status"] = "Yüz algılandı, ancak kayıtlı değil."
                response_data["name"] = "Bilinmeyen Kişi"
                response_data["confidence"] = round((1 - best_distance) * 100)

        # Netlify'e başarılı bir cevap (200) ve JSON verisini döndür
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Güvenlik için normalde buraya sitenizin adresi yazılır
            },
            'body': json.dumps(response_data)
        }

    except Exception as e:
        # Herhangi bir hata olursa, hatayı logla ve 500 hatası döndür
        print(f"Sunucu hatası: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Sunucuda bir hata oluştu.'})
        }
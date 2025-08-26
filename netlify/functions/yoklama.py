import json
import base64
import cv2
import numpy as np
import os
from deepface import DeepFace
import shutil

# DeepFace'in model dosyalarını indirebileceği ve kullanabileceği geçici bir alan oluşturma
# Netlify Functions'da sadece /tmp/ klasörü yazılabilirdir.
HOME = '/tmp/'
os.environ['DEEPFACE_HOME'] = HOME

def handler(event, context):
    try:
        body = json.loads(event['body'])
        image_data = body['image'].split(',')[1]

        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # DeepFace, yüzleri bir klasördeki resimlerle karşılaştırır.
        # Projemizdeki 'static/known_faces' klasörünü kullanacağız.
        db_path = "static/known_faces"

        # Yüz tanıma işlemini gerçekleştir
        # enforce_detection=False: Resimde yüz bulamazsa hata vermek yerine boş sonuç döner.
        dfs = DeepFace.find(
            img_path=img, 
            db_path=db_path, 
            model_name='VGG-Face', 
            enforce_detection=False
        )

        response_data = {
            "status": "Yüz algılandı, ancak kayıtlı değil.",
            "name": "Bilinmeyen Kişi",
            "confidence": None
        }

        # DeepFace bir sonuç bulduysa
        if dfs and not dfs[0].empty:
            # Sonuçlar bir "dataframe" içinde gelir, en iyi eşleşmeyi alalım.
            best_match = dfs[0].iloc[0]
            identity = best_match['identity']

            # Dosya yolundan ismi alalım (örn: static/known_faces/hasan/hasan1.jpg -> hasan)
            # Not: Windows'taki \ ayracını ve Unix'teki / ayracını hesaba katar
            name = os.path.basename(os.path.dirname(identity))

            response_data["status"] = f"Hoş geldin, {name.capitalize()}!"
            response_data["name"] = name.capitalize()
            # DeepFace doğrudan bir güven skoru vermez, ama bir eşleşme bulması yeterlidir.

        # Her çalıştırma sonrası /tmp klasörünü temizlemek iyi bir pratiktir.
        # Aksi takdirde bir sonraki çalıştırmada alan dolabilir.
        if os.path.exists(os.path.join(HOME, ".deepface")):
            shutil.rmtree(os.path.join(HOME, ".deepface"))

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }

    except Exception as e:
        print(f"Sunucu hatası: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Sunucuda bir hata oluştu.', 'details': str(e)})
        }
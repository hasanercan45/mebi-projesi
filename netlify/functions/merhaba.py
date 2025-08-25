import json

def handler(event, context):
    message = "Merhaba, dunya! Test fonksiyonu calisti."
    return {
        'statusCode': 200,
        'headers': { 'Content-Type': 'application/json' },
        'body': json.dumps({'message': message})
    }
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import boto3
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Initialize AWS Rekognition client
region = 'us-east-1'  # Replace 'your_region' with the actual AWS region
rekognition = boto3.client('rekognition', region_name=region)

@app.route('/detect', methods=['POST'])
def detect_helmet():
    try:
        # Get base64-encoded image data from request
        image_data = request.json['image']
        
        # Decode base64 image data and convert it to OpenCV format
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Call AWS Rekognition to detect labels in the image
        response = rekognition.detect_labels(Image={'Bytes': cv2.imencode('.jpg', image)[1].tobytes()}, MaxLabels=10)

        # Check if 'Helmet' label is detected
        helmet_detected = False
        for label in response['Labels']:
            if label['Name'] == 'Helmet':
                helmet_detected = True
                break
        
        return jsonify({'helmetDetected': helmet_detected}), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)  # Run Flask app in debug mode for development

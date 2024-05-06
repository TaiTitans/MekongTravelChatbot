import json
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from datetime import datetime, timedelta
import threading
import time
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from dotenv import load_dotenv
load_dotenv()
from flask_cors import CORS
app = Flask(__name__)
print("TensorFlow version:", tf.__version__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
# Kết nối đến MongoDB
import os
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo = PyMongo(app)

# Load mô hình vectorizer từ file .h5
vectorizer = joblib.load('vectorizer_model.h5')

# Load dữ liệu từ file JSON
with open('chatbot_data.json', 'r', encoding='utf-8') as file:
    chatbot_data = json.load(file)

questions = chatbot_data['questions']
answers = chatbot_data['answers']
question_vectors = chatbot_data['question_vectors']

# Class ChatbotMiddleware để xử lý logic của chatbot
class ChatbotMiddleware:
    def __init__(self):
        self.vectorizer = vectorizer
        self.questions = questions
        self.answers = answers
        self.question_vectors = question_vectors

    def predict_intent(self, message):
        input_vector = self.vectorizer.transform([message])
        similarities = cosine_similarity(input_vector, self.question_vectors)
        most_similar_index = similarities.argmax()
        return int(most_similar_index)  # Chuyển thành kiểu int

    def generate_response(self, intent_id):
        response = self.answers[intent_id]
        return response

# Khởi tạo đối tượng ChatbotMiddleware
chatbot_middleware = ChatbotMiddleware()

# Model trong Flask
class Chat:
    def __init__(self, message, intent_id):
        self.message = message
        self.intent_id = intent_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def save(self):
        mongo.db.chats.insert_one({
            'message': self.message,
            'intent_id': self.intent_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        })

# Xoá các cuộc trò chuyện cũ khi đạt đến ngưỡng
def delete_old_chats():
    try:
        chat_threshold = 10  # Ngưỡng số lượng cuộc trò chuyện
        current_chat_count = mongo.db.chats.count_documents({})
        if current_chat_count > chat_threshold:
            oldest_chats = mongo.db.chats.find().sort([('created_at', 1)]).limit(current_chat_count - chat_threshold)
            for chat in oldest_chats:
                mongo.db.chats.delete_one({'_id': chat['_id']})
            print('Deleted old chats successfully.')
    except Exception as e:
        print('Error deleting old chats:', e)

# Lên lịch xoá tự động
def schedule_delete():
    while True:
        now = datetime.now()
        # Xác định thời điểm bắt đầu của ngày mới (00:00:00)
        tomorrow_start = datetime(now.year, now.month, now.day) + timedelta(days=1)
        # Tính thời gian đợi đến thời điểm bắt đầu của ngày mới
        wait_time = (tomorrow_start - now).total_seconds()
        time.sleep(wait_time)
        delete_old_chats()

# Khởi tạo worker cho việc xoá tự động
schedule_thread = threading.Thread(target=schedule_delete)
schedule_thread.start()

# Controller trong Flask
@app.route('/create_chat', methods=['POST'])
def create_chat():
    try:
        data = request.get_json()
        message = data['message']

        # Dự đoán intent từ tin nhắn
        intent_id = chatbot_middleware.predict_intent(message)

        # Tạo một cuộc trò chuyện mới
        new_chat = Chat(message, intent_id)
        new_chat.save()

        # Tạo câu trả lời dựa trên intentId
        response = chatbot_middleware.generate_response(intent_id)

        return jsonify({'chat': new_chat.__dict__, 'response': response})
    except Exception as e:
        print('Error creating chat:', e)
        return jsonify({'error': 'Có lỗi xảy ra. Vui lòng thử lại sau.'}), 500

# Router
if __name__ == '__main__':
    app.run(debug=True)

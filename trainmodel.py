import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Đọc dữ liệu từ file JSON
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Tách dữ liệu thành câu hỏi và câu trả lời
questions = []
answers = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        questions.append(pattern)
    responses = intent['responses']
    for pattern in intent['patterns']:
        answer = responses[questions.index(pattern) % len(responses)]
        answers.append(answer)

# Tạo TF-IDF vector cho câu hỏi
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Lưu mô hình vectorizer thành file H5
joblib.dump(vectorizer, 'vectorizer_model.h5')

# Lưu dữ liệu thành file JSON
output_data = {
    'questions': questions,
    'answers': answers,
    'question_vectors': question_vectors.toarray().tolist()
}

with open('chatbot_data.json', 'w', encoding='utf-8') as outfile:
    json.dump(output_data, outfile, ensure_ascii=False)

# Hàm trả về câu trả lời dựa trên sự tương đồng với câu hỏi
def get_response(user_input):
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    most_similar_index = similarities.argmax()
    return answers[most_similar_index]

# Ví dụ sử dụng
user_input = "Tạm biệt"
response = get_response(user_input)
print(response)

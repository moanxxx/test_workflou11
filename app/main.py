from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr
import json

# Шаг 1: Загрузка JSON-документации
def load_json_to_text(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    documents = []
    for key, value in data.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, indent=4)
        text = f"{key}: {value}"
        documents.append(text)
    return documents

# Укажите путь к вашему JSON-файлу
json_path = "rest_api_documentation.json"
documents = load_json_to_text(json_path)

# Шаг 2: Загрузка локальной модели эмбеддингов
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# Шаг 3: Загрузка модели генерации текста
model_name = "tiiuae/falcon-7b-instruct"  # Замените на нужную модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Автоматически распределяет модель на доступные устройства
    trust_remote_code=True
)

# Шаг 4: Функция для поиска ответа
def find_relevant_document(question):
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(question_embedding, doc_embeddings)
    closest_idx = np.argmax(similarities)
    return documents[closest_idx]

# Шаг 5: Генерация текста с использованием модели
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=500,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Шаг 6: Интеграция поиска и генерации
def ask_bot(question):
    relevant_doc = find_relevant_document(question)
    prompt = f"Документация: {relevant_doc}\nВопрос: {question}\nОтвет:"
    return generate_text(prompt)

# Шаг 7: Интерфейс пользователя
interface = gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs="text",
    title="ИИ-консультант по JSON-документации"
)
interface.launch()

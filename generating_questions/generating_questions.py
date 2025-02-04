
import spacy
import PyPDF2
import json
import time
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

nlp = spacy.load("ru_core_news_sm")

giga = GigaChat(credentials="N2JkNjNkNzgtOWU2NC00OTVkLTk0MGItZWJjY2Y0Mjg2YTE0OjgzODlhODdiLTk4Y2ItNGJhOC04NjIyLWJlNDYxOTVlYzc1Yg==", verify_ssl_certs=False)


def read_pdf_file(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text()).strip()
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def extract_key_sentences(doc, num_sentences=5):
    sentences = [sent.text.strip() for sent in doc.sents if len(sent) > 10]
    return sentences[:num_sentences]


def generate_test_statements(text):
    doc = nlp(text)
    key_sentences = extract_key_sentences(doc, num_sentences=5)

    test_statements = []

    for sentence in key_sentences:
        print(f"\nГенерация утверждения для: {sentence}")

        system_message = Messages(
            role=MessagesRole.SYSTEM,
            content=(
                "Создай тестовое утверждение с пропуском `____`, где пропущенное слово является правильным ответом.\n\n"
                "**Формат вывода (строго JSON, без лишнего текста!):**\n"
                "{\n"
                '  "statement": "Текст утверждения с `____`",\n'
                '  "correct": "Правильный ответ",\n'
                '  "incorrect": ["Неправильный 1", "Неправильный 2", "Неправильный 3"]\n'
                "}\n\n"
                "**Требования:**\n"
                "- Утверждение должно быть ПОЛНЫМ и понятным без дополнительных пояснений.\n"
                "- Пропущенное слово ДОЛЖНО быть в конце утверждения.\n"
                "- НЕ начинай утверждение с `____`.\n"
                "- Формат вывода строго в JSON без комментариев.\n\n"
                "**Текст, на основе которого нужно создать утверждение:**\n"
                f"{sentence}"
            )
        )

        payload = Chat(messages=[system_message], temperature=0.1, max_tokens=150)
        response = giga.chat(payload)

        generated_json = response.choices[0].message.content.strip()

        print(f"Отладка: {generated_json}") 

        try:
            question_data = json.loads(generated_json)
            test_statements.append(question_data)  
        except json.JSONDecodeError:
            print("Пропуск вопроса.")

        time.sleep(0.2)  

    return test_statements


source_text = read_pdf_file("лекция.pdf")

if source_text:
    test_statements = generate_test_statements(source_text)

    json_filename = "test_statements.json"
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(test_statements, json_file, ensure_ascii=False, indent=4)

    print(f"\nСохранение в файл {json_filename}")
else:
    print("Ошибка текста.")
# Example 4.24 Modified: Question Answering with Custom Context and Questions
from transformers import pipeline

# Inisialisasi pipeline question-answering
question_answerer = pipeline('question-answering')

# Menentukan teks konteks baru
context_text = """
Politeknik Elektronika Negeri Surabaya (PENS) is a vocational college located in Surabaya, Indonesia. 
Established in 1988, PENS is widely recognized as a center of excellence in electronics, multimedia engineering, 
and artificial intelligence research. Students frequently work on computer vision projects using Python and TensorFlow.
"""

# Daftar pertanyaan yang diajukan terhadap konteks
questions = [
    "Where is PENS located?",
    "When was PENS established?",
    "What are the main areas of research at PENS?",
    "What programming tools do students use for projects?",
]

# Menjalankan inferensi untuk setiap pertanyaan
for q in questions:
    result = question_answerer({'question': q, 'context': context_text})
    print(f"Question : {q}")
    print(
        f"Answer   : {result['answer']} (Score: {result['score']:.4f})\n"
    )
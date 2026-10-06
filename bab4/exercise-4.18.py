# Example 4.22 Modified: Custom Text Analysis using NLTK
# https://www.nltk.org/
# pip install nltk
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Mengganti teks masukan dengan narasi baru
sentence = """
Surabaya is the second-largest city in Indonesia and a major hub for technology education.
Students at Politeknik Elektronika Negeri Surabaya frequently develop advanced artificial intelligence applications,
ranging from computer vision systems to mixed reality experiences using Python and TensorFlow.
"""

tokens = nltk.word_tokenize(sentence)
print("Tokens:", tokens)

tagged = nltk.pos_tag(tokens)
print("POS Tagging:", tagged)

entities = nltk.chunk.ne_chunk(tagged)
print("Named Entities:")
print(entities)
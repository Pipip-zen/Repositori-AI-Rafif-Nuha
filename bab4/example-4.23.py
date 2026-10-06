# Example 4.23
# https://github.com/huggingface/transformers
# Install model support in this environment: pip install transformers torch
# https://aka.ms/vs/16/release/vc_redist.x64.exe
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('This is a good movie.'))
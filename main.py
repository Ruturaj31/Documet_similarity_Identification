#import px as px
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pyttsx3
import PyPDF2

# Define the preprocess function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return set(filtered_tokens)
import io

# Function to extract text from PDF file
def extract_text_from_pdf(file_upload):
    pdf_text = ""
    with io.BytesIO(file_upload.getvalue()) as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            pdf_text += pdf_reader.getPage(page_num).extractText()
    return pdf_text
# Define Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Define cosine similarity function
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Define a function to read text with pyttsx3
def read_text(text):
    pdf_speaker = pyttsx3.init()
    voices = pdf_speaker.getProperty('voices')
    for voice in voices:
        pdf_speaker.setProperty('voice', voice.id)
    pdf_speaker.say(text)
    pdf_speaker.runAndWait()

# Streamlit app
st.set_page_config(
    page_title="Text Similarity and Text-to-Speech App 📚",
    page_icon="📚",
    layout="wide",
)

# Add HTML and CSS for background animation
st.markdown(
    """
    <style>
    body {
        background-color: #393929;
        background-image: url('C:/Users/HP/PycharmProject/ject/Coming Soon Website Coming Soon Page in Black White Dark Futuristic Style.jpg');
        background-size: cover;
        background-blur: 15px;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app content
st.title("Text Similarity and Text-to-Speech App")
# Input text or files for similarity
input_doc1 = st.text_area("Enter the first document:")
input_doc2 = st.text_area("Enter the second document:")
file_upload_1 = st.file_uploader("Upload the first PDF file for comparison", type=["pdf"])
file_upload_2 = st.file_uploader("Upload the second PDF file for comparison", type=["pdf"])

pdf_text_1 = ""
pdf_text_2 = ""

if file_upload_1 is not None and file_upload_2 is not None:
    # Extract text from the uploaded PDF files
    pdf_text_1 = extract_text_from_pdf(file_upload_1)
    pdf_text_2 = extract_text_from_pdf(file_upload_2)

# Calculate Similarity Button
if st.button("Calculate Similarity"):
    if file_upload_1 is not None and file_upload_2 is not None:
        # Use the extracted text from the PDF files for similarity comparison
        input_doc1 = pdf_text_1
        input_doc2 = pdf_text_2

    set1 = preprocess(input_doc1)
    set2 = preprocess(input_doc2)

    # Check if both sets are empty to avoid ZeroDivisionError
    if not set1 and not set2:
        st.warning("Both documents are empty.")
        st.stop()

    # Convert the documents to vectors using Bag-of-Words model
    all_words = list(set(list(set1) + list(set2)))
    vec1 = [int(word in set1) for word in all_words]
    vec2 = [int(word in set2) for word in all_words]

    # Calculate Jaccard similarity
    jaccard = jaccard_similarity(set1, set2)

    # Calculate cosine similarity
    cosine = cosine_similarity(vec1, vec2)

    # Calculate average similarity
    similarity = (jaccard + cosine) / 2

    threshold = 0.7

    result = "The Documents are similar" if similarity >= threshold else "The Documents are not Similar"

    st.write(f"Jaccard similarity: {jaccard:.2f}")
    st.write(f"Cosine similarity: {cosine:.2f}")
    st.write(f"Average similarity: {similarity:.2f}")
    st.write(result)
    # Bar Chart Visualization
    categories = ['Jaccard Similarity', 'Cosine Similarity', 'Average Similarity']
    values = [jaccard, cosine, similarity]



    plt.bar(categories, values)
    plt.xlabel('Similarity Metric')
    plt.ylabel('Similarity Value')
    plt.title('Document Similarity Metrics')
    st.pyplot()


# Input text for text-to-speech
text_to_speech_input = st.text_area("Enter text for Text-to-Speech:")

if st.button("Play Text-to-Speech"):
    read_text(text_to_speech_input)
st.set_option('deprecation.showPyplotGlobalUse', False)



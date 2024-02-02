import re
import string
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

def pie_chart(text):
    h = pd.DataFrame({'text':text.split()})
    j = pd.DataFrame(h.text.value_counts(normalize=True))
    plt.pie(j.text, labels=j.index, autopct='%1.1f%%', startangle=90)
    pie_labels = plt.gca().patches
    categories_from_pie = [label.get_label() for label in pie_labels]    
    return categories_from_pie

def bar_plot(text):
    h = pd.DataFrame({'text':text.split()})
    j = pd.DataFrame(h.text.value_counts())
    hj = plt.bar(j.index, j.text, color='blue')
    ax = plt.gca()
    categories_from_plot = [tick.get_text() for tick in ax.get_xticklabels()]  
    return categories_from_plot, hj.datavalues

def cloud_words(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  
    plt.show()
    return wordcloud

def plot_word_frequency(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()

    word_counts = Counter(words)
    
    common_words = word_counts.most_common(10) 
    
    words, counts = zip(*common_words)
    
    # Crear el gráfico de barras
    plt.bar(words, counts)
    plt.xlabel('Palabra')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de Palabras')
    return plt.gcf()

def tokenize_text(text):
    return word_tokenize(text)

def remove_characters(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    else:
        return text

def remove_stopwords(text):
    if isinstance(text, list):
        stop_words = set(stopwords.words('english'))
        return [word for word in text if word.lower() not in stop_words]
    
def remove_stopwords2(text):
    if not isinstance(text, list):
        raise ValueError("Input should be a list of words")
    
    stop_words = set(stopwords.words('english'))
    return [word for word in text if word.lower() not in stop_words]

def perform_stemming(text):
    if isinstance(text, list):
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in text]

def perform_lemmatization(text):
    if isinstance(text, list):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in text]

def convert_to_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    else:
        return text

def remove_emojis(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text

def remove_extra_spaces(text):
    if isinstance(text, str):
        return re.sub(' +', ' ', text)
    else:   
        return text

def remove_numbers(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    else:
        return text

def remove_users(text):
    if isinstance(text, str):
        return re.sub(r'@\S+', '', text)
    else:
        return text
    
def remove_hastags(text):
    if isinstance(text, str):
        return re.sub(r'#\S+', '', text)
    else:
        return text

def remove_links(text):
    if isinstance(text, str):
        return re.sub(r'http\S+', '', text)
    else:
        return text

def transform_stemmer(text):
    if isinstance(text, str):
        # Initialize Python porter stemmer
        ps = PorterStemmer()

        # Remove punctuation
        example_sentence_no_punct = text.translate(str.maketrans("", "", string.punctuation))
        
        # Create tokens
        word_tokens = word_tokenize(example_sentence_no_punct)

        # Perform stemming
        return ' '.join([ps.stem(word) for word in word_tokens])
    else:
        return text

def transform_lemmatizer(text):
    if isinstance(text, str):
        # Initialize wordnet lemmatizer
        wnl = WordNetLemmatizer()

        # Remove punctuation
        example_sentence_no_punct = text.translate(str.maketrans("", "", string.punctuation))
        
        word_tokens = word_tokenize(example_sentence_no_punct)

        # Perform lemmatization
        return ' '.join([wnl.lemmatize(word) for word in word_tokens ])
    else:
        return text

def preprocess_text(text):
    text = remove_links(text)
    text =remove_hastags(text)
    text = remove_characters(text)
    text =tokenize_text(text)
    text = remove_stopwords(text)
    text = perform_stemming(text)
    text = perform_lemmatization(text)
    text = remove_emojis(text)
    text = remove_extra_spaces(text)
    text = remove_numbers(text)
    return ' '.join(text)
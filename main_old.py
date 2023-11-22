import nltk
import csv
import numpy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

def calculate_sim(mat, text):
    cosine_sim = cosine_similarity(mat, mat)

    # Example: Recommend books similar to book 0 (change index as needed)
    book_index = len(text) - 1
    similar_books = list(enumerate(cosine_sim[book_index]))

    # Sort books by similarity
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

    # Display top similar books
    for book in similar_books[:6]:  # Change the number to show more or fewer recommendations
        if book[0] == book_index:
            continue
        print(f"Book Title: {titles[book[0]]}, Similarity: {book[1]}")



def recommend_books(text):
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)

    sentiments = [TextBlob(element).sentiment.polarity for element in text]
    # Feature engineering: Text Length
    text_lengths = [len(element.split()) for element in text]

    # Convert the lists to numpy arrays
    sentiments = numpy.array(sentiments).reshape(-1, 1)
    text_lengths = numpy.array(text_lengths).reshape(-1, 1)

    # Concatenate the features with the TF-IDF matrix
    tfidf_sent = numpy.hstack((tfidf_matrix.toarray(), sentiments))
    tfidf_features = numpy.hstack((tfidf_matrix.toarray(), sentiments, text_lengths))

    print('No sentiment')
    calculate_sim(tfidf_matrix, text)
    print('Sentiment')
    calculate_sim(tfidf_sent, text)
    print('Sentiment+length')
    calculate_sim(tfidf_features, text)


# Open the CSV file
with open('book_data.csv', 'r', encoding='utf-8') as file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(file)

    tfidf_vectorizer = TfidfVectorizer()
    descriptions = []
    processed_descriptions = []
    titles=[]
    # Iterate over each row in the CSV file
    for row in csv_reader:
        desc = row['Description']
        
        titles.append(row['Title'])
        
        desc = desc.lower()
        desc = desc.replace(".", "")
        desc = desc.replace(",", "")
        desc = desc.replace("'", "")
        desc = desc.replace('"', "")
        desc = desc.replace("(", "")
        desc = desc.replace(")", "")
        desc = desc.replace("[", "")
        desc = desc.replace("]", "")

        stopwords_set = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()
        porter = PorterStemmer()

        tokens = nltk.word_tokenize(desc)

        tokens = [word for word in tokens if word not in stopwords_set]

        descriptions.append(tokens)

        halfprocessed_texts = [' '.join(tokens) for tokens in descriptions]

        tokens = [wnl.lemmatize(t) for t in tokens]

        tokens = [porter.stem(t) for t in tokens]

        processed_descriptions.append(tokens)

        preprocessed_texts = [' '.join(tokens) for tokens in processed_descriptions]

print('Half-processed')
print('-----------')
recommend_books(halfprocessed_texts)
print('Processed')
print('-----------')
recommend_books(preprocessed_texts)

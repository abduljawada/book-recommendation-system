import csv
import gensim
from nltk.corpus import stopwords
import collections
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer, PorterStemmer

input = 'Set on the desert planet Arrakis across the known universe, melange is a prize worth killing for... When House Atreides is betrayed, the destruction of Paul’s family will set the boy on a journey toward a destiny greater than he could ever have imagined. And as he evolves into the mysterious man known as Muad’Dib, he will bring to fruition humankind’s most ancient and unattainable dream.'
stopwords_set = set(stopwords.words('english'))
wnl = WordNetLemmatizer()
porter = PorterStemmer()

titles = []
descs = []

max_length = 90  # Set your desired maximum length

def read_corpus():
    # Open the CSV file
    with open('book_data.csv', 'r', encoding='utf-8') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        

     
        # List to store tokenized and preprocessed documents
        tokenized_documents = []

        for i, line in enumerate(csv_reader):
            tokens = gensim.utils.simple_preprocess(line['Description'])

            descs.append(line['Description'])

            tokens = [word for word in tokens if word not in stopwords_set]
            tokens = [wnl.lemmatize(t) for t in tokens]
            tokens = [porter.stem(t) for t in tokens]

            titles.append(line['Title'])

            # Append tokenized and preprocessed document to the list
            tokenized_documents.append(tokens)

        # After processing all documents, pad sequences to a maximum length
       
        global padded_documents
        padded_documents = pad_sequences(tokenized_documents, dtype=object, maxlen=max_length, value='_PAD_')

        for j, jline in enumerate(padded_documents):
            yield gensim.models.doc2vec.TaggedDocument(jline, [j])



train_corpus = list(read_corpus())

model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=3, epochs=300)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

ranks = []

for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims[:5]]
    print(rank)
    for num in rank: 
        ranks.append(num)


counter = collections.Counter(ranks)
print(counter)

tokenized_input = gensim.utils.simple_preprocess(input)
tokenized_input = [word for word in tokenized_input if word not in stopwords_set]
tokenized_input = [wnl.lemmatize(t) for t in tokenized_input]
tokenized_input = [porter.stem(t) for t in tokenized_input]

doc_tokens = []

doc_tokens.append(tokenized_input)

padded_documents = pad_sequences(doc_tokens, dtype=object, maxlen=max_length, value='_PAD_')

print(tokenized_input)

vector = model.infer_vector(tokenized_input)

sims = model.dv.most_similar([vector], topn=len(model.dv))
for book in sims[:5]:  # Change the number to show more or fewer recommendations
        print(titles[book[0]], '\n', descs[book[0]], '\n')
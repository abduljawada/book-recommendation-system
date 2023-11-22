import csv
import gensim
from nltk.corpus import stopwords

#Veronika decides to die, tells the story of veronika who after feeling like her life is not interesting but then is rescued and moved to an asylum where she is told she only has one month to live and discovers there the value of her life

input = 'Veronika decides to die, tells the story of veronika who after feeling like her life is not interesting but then is rescued and moved to an asylum where she is told she only has one month to live and discovers there the value of her life'

titles = []
descs = []
        
stopwords_set = set(stopwords.words('english'))

def read_corpus():
    # Open the CSV file
    with open('book_data.csv', 'r', encoding='utf-8') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
     
        for i, line in enumerate(csv_reader):
            tokens = gensim.utils.simple_preprocess(line['Description'])

            descs.append(line['Description'])

            tokens = [word for word in tokens if word not in stopwords_set]

            titles.append(line['Title'])
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus())
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

tokenized_input = gensim.utils.simple_preprocess(input)
tokenized_input = [word for word in tokenized_input if word not in stopwords_set]

print(tokenized_input)

vector = model.infer_vector(tokenized_input)

sims = model.dv.most_similar([vector], topn=len(model.dv))
for book in sims[:5]:  # Change the number to show more or fewer recommendations
        print(titles[book[0]], '\n', descs[book[0]], '\n')

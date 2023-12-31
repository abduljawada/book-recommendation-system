from textblob import TextBlob

text = "Newspeak, Doublethink, Big Brother, the Thought Police - the language of 1984 has passed into the English language as a symbol of the horrors of totalitarianism. George Orwell's story of Winston Smith's fight against the all-pervading Party has become a classic, not the least because of its intellectual coherence. First published in 1949, it retains as much relevance today as it had then."

# Create a TextBlob object
blob = TextBlob(text)

# Get the sentiment polarity score
polarity_score = blob.sentiment.polarity

print(f"Polarity Score: {polarity_score}")

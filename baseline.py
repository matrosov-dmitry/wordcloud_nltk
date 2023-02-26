import nltk
import pymorphy2

from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')


# открытие файла

with open('speech.txt', 'r') as file:
    content = file.read()

# токенизация 

words = word_tokenize(content)


word_no_punc = []

for word in words:
    if word.isalpha():
        word_no_punc.append(word.lower())

stopwords_list = stopwords.words('russian')

clean_words = []

for word in word_no_punc:
    if word not in stopwords_list:
        clean_words.append(word)

snowball = SnowballStemmer(language='russian')

stem_words = []

for word in clean_words:
    stem_words.append(snowball.stem(word))

morph = pymorphy2.MorphAnalyzer()

lemm_words = []

for word in stem_words:
    lemm_words.append(morph.parse(word)[0].normal_form)

word_to_cloud = " ".join(lemm_words)

wordcloud = WordCloud(background_color='white',
                      max_words=30).generate(word_to_cloud)

plt.figure()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
                      

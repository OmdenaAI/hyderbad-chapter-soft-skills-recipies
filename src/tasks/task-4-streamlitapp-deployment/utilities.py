from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
#import nltk_download_utils
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import wordnet 
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('averaged_perceptron_tagger')
import string

def page_data(result_df,from_i,page_size,current_page):
  end=current_page*page_size
  result=result_df[from_i:end]
  return result


def search_words_phrases(search):
  regex=''
  ds_l=search.split(' ')
  for elem in ds_l:
    regex+=f'(?=.*{elem})' 
  return regex


def punctuation_stop_word_removal_and_tokenization(input_str):
  tokens=[]
  content=[]
  content.append(input_str)
  #tokenize in lower case
  for item in content:
      tokens.extend([word.lower() for word in nltk.word_tokenize(item)])
  #remove stopwords
  tokens_without_sw = [word for word in tokens if not word in stopwords.words()]
  words_no_punc=[]
  for word in tokens_without_sw:
    if word.isalpha():
        words_no_punc.append(word.lower())
  return words_no_punc


def word_cloud(result_df):
  text = " ".join(i for i in result_df)
  tokens_without_sw=punctuation_stop_word_removal_and_tokenization(text)
  lemmas=lemmantization(tokens_without_sw)
  # print(lemmas)
  input=' '.join(lemmas)
  # print(input)
  wordcloud = WordCloud(background_color="white", colormap='Greys').generate(input)
  fig, ax = plt.subplots(figsize=(27,7))
  ax.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  return fig



def lemmantization(tokens_without_sw):
  lemmas = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in tokens_without_sw]
  return lemmas


#pos tagging
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)



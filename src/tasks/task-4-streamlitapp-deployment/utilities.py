from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

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


def word_cloud(result_df):
  text = " ".join(i for i in result_df.Summary)
  stopwords = set(STOPWORDS)
  wordcloud = WordCloud(stopwords=stopwords, background_color="white", colormap='Greys').generate(text)
  fig, ax = plt.subplots(figsize=(15,10))
  ax.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  return fig
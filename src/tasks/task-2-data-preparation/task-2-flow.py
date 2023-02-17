!pip install sentence-transformers

# imports
import pandas as pd
import sklearn.feature_extraction.text as txt
import bs4
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import time
import transformers
from transformers import pipeline

# function to read data and verify it's structure
def verify_data_structure(df):
    
    cols = set(['Soft Skill Name', 'Criteria', 'URL', 'Title of URL', 'Content'])
    
    if not isinstance(df, pd.DataFrame):
        raise Exception("df must be a dataframe that contains these columns : " + str(cols))
        
    if cols.issubset(set(list(df.columns))):
        return df
    else:
        raise Exception("incorrect dataframe structure ! - expect these columns : " + str(cols))
        
def read_data(path_):
    data = pd.read_csv(path_)
    
    # verify data is in the required form
    return verify_data_structure(data)        

def pre_processing_data(data, min_paragraphs_words = 10):
    
    print('----> data pre-processing - initial shape of data : ', data.shape)
    
    # clean paragraph
    data['Title of URL'] = data['Title of URL'].apply(lambda x : re.sub("[\s\n\t\b\']+"," ", str(x)).strip())

    # remove duplicate URLs
    data = data.drop_duplicates('URL')
    
    print('----> data pre-processing - shape of data after duplicate removal : ', data.shape)

    # length of title and content
    data['length_of_title'] = data['Title of URL'].apply(len)
    data['length_of_content'] = data['Content'].apply(len)

    # number of words (space separeted)
    def n_words(text):
        return len(text.split(' '))

    data['n_words_title'] = data['Title of URL'].apply(n_words)
    data['n_words_content'] = data['Content'].apply(n_words)

  # extract paragraphs from Content
    def extract_paragraphs(content):
        paragraph = ''
        soup = bs4.BeautifulSoup(content, "html.parser")
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            paragraph = paragraph + p.get_text()
        return paragraph

    data['paragraphs'] = data['Content'].apply(extract_paragraphs)
    data['length_of_paragraphs'] = data['paragraphs'].apply(len)

    # clean paragraphs
    data['paragraphs'] = data['paragraphs'].apply(lambda x : re.sub("[\s\n\t\b\']+"," ", str(x)).strip())
    
    # number of words in paragraphs
    data['n_words_paragraphs'] = data['paragraphs'].apply(n_words)
    
    # take paragraphs that has a min number of words
    data = data.loc[data['n_words_paragraphs'] > min_paragraphs_words , :]
    
    print('----> data pre-processing - shape of data after min words paragraphs removal : ', data.shape)

    #Stats number of words
    data[['n_words_title', 'n_words_paragraphs']].describe().T

    return data


# search top n most frequent words
def get_top_n_frequent_word(content, n = 1):
    
    stop_words_ = txt.ENGLISH_STOP_WORDS
    count_vect = CountVectorizer(stop_words = 'english') # TODO : test ...stop_words = stop_words_)
    
    # get the matrix for words count
    corpus = [content] # corpus must be a list that's why we do this
    try:
        X_words_counts = count_vect.fit_transform(corpus)
    except :
        return '', 0

    words = []
    counts = []
    for word in count_vect.vocabulary_:
        try:
            i = count_vect.vocabulary_[word] # get the index of the word in the matrix of words
            words.append(word)
            counts.append(X_words_counts[0,i])
        except:
            words.append('')
            counts.append(0)
            continue
            

    counts_df = pd.DataFrame.from_dict({'word' : words, 'count':counts})
    counts_df = counts_df.sort_values(by='count', ascending = False)
    counts_df = counts_df.head(n).reset_index().drop('index', axis=1)
    
    words = list(counts_df['word'])
    counts = list(counts_df['count'])
    
    return words[-1], counts[-1]


# most frequent word 
def get_most_frequent_word(content):
    return get_top_n_frequent_word(content, 1)[0]

def get_most_frequent_word_count(content):
    return get_top_n_frequent_word(content, 1)[1]
    

def words_frequency(data, col = 'paragraphs'):
    data[col + '_most_frequent_word'] = data[col].apply(get_most_frequent_word)
    data[col + '_most_frequent_word_count'] = data[col].apply(get_most_frequent_word_count)

    return data



# this function disaggregates the content column, which is a big set of paragraphs into 
# many smaller paragraphs

#**************************************************************************************
# to save memory the Content and paragraphs will be droped from the returned dataframe
#**************************************************************************************
            
def disaggregate_content_medium(df): 
    # these list are used to constract the returned dataframe
    soft_kill_names_df = [] # 
    criterias_df = [] 
    URLs_df = []
    titles_of_URLs_df = []
    summaries_df = [] # summary produced in the previous step (summary of all the concateneted paragraphs)
    paragraphs_df = []
    headings_df = [] # heading just before paragraph
    headings_type_df = [] # type of tytle : h1, h2,...
    

    for idx, row in df.iterrows():
        
        paragraphs = row['Content'].split('\n')
        
        for paragraph in paragraphs:
            
            
            if paragraph == '': # if empty we continue
                continue
            # get the old columns of df except Content beacause it's huge and we won't need it later
            soft_kill_names_df.append(row['Soft Skill Name']) 
            criterias_df.append(row['Criteria']) 
            URLs_df.append(row['URL'])
            titles_of_URLs_df.append(row['Title of URL'])
            summaries_df.append('no summary')#summaries_df.append(row['summary'])
        
    disaggregated_df = pd.DataFrame.from_dict({'Soft Skill Name' : soft_kill_names_df, 
                                               'Criteria' : criterias_df, 
                                               'URL': URLs_df, 
                                               'Title of URL' : titles_of_URLs_df, 
                                               'summary of Content' : summaries_df,
                                               'paragraph' : paragraphs_df})
    disaggregated_df['header'] = ''
    return disaggregated_df




# this function disaggregates the content column (html format) of the df dataframe into many paragraphs
# For each paragraph, it searchs for the first previous heading found

#**************************************************************************************
# to save memory the Content and paragraphs will be droped from the returned dataframe
#**************************************************************************************

def disaggregate_content(df, tag = 'p', a_class = None): # df is supposed a dataframe that contains the Content column in the html format : <body>....</body>
    # these list are used to constract the returned dataframe
    soft_kill_names_df = [] # 
    criterias_df = [] 
    URLs_df = []
    titles_of_URLs_df = []
    summaries_df = [] # summary produced in the previous step (summary of all the concateneted paragraphs)
    paragraphs_df = []
    headings_df = [] # heading just before paragraph
    headings_type_df = [] # type of tytle : h1, h2,...
    

    for idx, row in df.iterrows():
        soup = bs4.BeautifulSoup(row['Content'], "html.parser")
        
        if a_class != None:
            paragraphs = soup.find_all(tag, a_class) 
        else:
            paragraphs = soup.find_all(tag) 
            
        for paragraph in paragraphs:
            
            # get paragraph text
            temp = paragraph.get_text()
            if temp == '': # if empty we continue
                continue
            
            paragraphs_df.append(paragraph.get_text())
            
            # get the old columns of df except Content beacause it's huge and we won't need it later
            soft_kill_names_df.append(row['Soft Skill Name']) 
            criterias_df.append(row['Criteria']) 
            URLs_df.append(row['URL'])
            titles_of_URLs_df.append(row['Title of URL'])
            summaries_df.append('no summary')#summaries_df.append(row['summary'])
            
            # find heading just before that paragraph. find_heading returns (title, hx). See definition below
            heading = find_heading(paragraph)
            headings_df.append(heading[0]) # the text heading
            try:
                # it's type : h1...h6
                headings_type_df.append(heading[1])
            except:
                # if no heading found affect None
                headings_type_df.append('None')
                continue
        
    
    disaggregated_df = pd.DataFrame.from_dict({'Soft Skill Name' : soft_kill_names_df, 
                                               'Criteria' : criterias_df, 
                                               'URL': URLs_df, 
                                               'Title of URL' : titles_of_URLs_df, 
                                               'summary of Content' : summaries_df,
                                               'header':headings_df,
                                                'paragraph' : paragraphs_df})
    
    
    # clean paragraph
    disaggregated_df['paragraph'] = disaggregated_df['paragraph'].apply(lambda x : re.sub("[\s\n\t\b\']+"," ", str(x)).strip())

    disaggregated_df = disaggregated_df.drop_duplicates('paragraph')
    
    return disaggregated_df



# this function looks for the first heading previous to elt passed as parameter
# starting form h6 until h1
def find_heading(elt):
    headings = ['h6', 'h5', 'h4', 'h3', 'h2', 'h1']
    elt = elt.previous_element
    while (not (elt is None) ) and (not (elt.name in headings)):
         elt = elt.previous_element
    
    if (not (elt is None) ):
        return elt.get_text(), elt.name
    else:
        return 'None', 'None'



#Cosine similarity function
def cos_sim(p1, p2, model):
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    p1_embed = model.encode(p1)
    p2_embed = model.encode(p2)
    similarity = cosine_similarity([p1_embed], [p2_embed])
    return similarity[0]


#Calculate similarity with respect to a column
def paragraph_similarity_col(dataframe, col, model):
    
    assert col in list(dataframe.columns), col + ' not present in dataframe !'
    
    url_col = dataframe[col].unique()                                                                
    similarity = []                                              

    for i in range(0, len(url_col)):
        content_url_col = dataframe[dataframe[col] == url_col[i]]['paragraph']           
        idx = content_url_col.index                                                                    

        for j in idx:                                                                  
            sim = cos_sim(url_col[i], content_url_col[j], model)[0]                  
            similarity.append(sim)

    dataframe['similarity_wrt_' + col] = similarity

    return dataframe

#calculate similarity with respect to reference paragraph
def similarity_wrt_ref_par(dataframe, model):
    url_titles = dataframe['Title of URL'].unique() 
    simi = []                                                                     

    for i in range(0, len(url_titles)):
        dft = dataframe[dataframe['Title of URL'] == url_titles[i]]                
        max_similar = dft['similarity_wrt_title'].max()                            
        ref_para = dft[dft['similarity_wrt_title'] == max_similar]['paragraph']    
        ref_index = ref_para.index[0]                                        
        paragraph = dft['paragraph']
        idx = paragraph.index

    for i in idx:
        sim = cos_sim(ref_para[ref_index], paragraph[i], model)[0]                       
        simi.append(sim)

    dataframe['similarity_wrt_ref_para'] = simi

    return dataframe



#calculate the difference between two consecutive paragraphs
def difference(dataframe, diff_treshold = 0.05):
    url_titles = dataframe['Title of URL'].unique()
    index = []                                                 

    for i in range(0, len(url_titles)):
        dft = dataframe[dataframe['Title of URL'] == url_titles[i]]                 
        simil = dft['similarity_wrt_ref_para']
        idx = simil.index

    for i in range(0, len(idx) - 2):                                    
        diff = abs(simil[idx[i]] - simil[idx[i+1]])                       
        if diff <  diff_treshold:
            index.append(idx[i+1])    

    return dataframe.drop(indices, axis = 0)

  

def zs_classify(data, candidate_labels):

    text_list = list(data['paragraph'])

    zeroshot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    labels = []
    scores = []
    for t in text_list:
        class_ = zeroshot_classifier(t, candidate_labels, multi_label=False)
        labels.append(class_['labels'][0])
        scores.append(class_['scores'][0])

    data['zs_class'] = labels
    data['zs_class_score'] = scores

    return data

def paragraph_stats(dataframe):
    dataframe['n_words_paragraph'] = dataframe['paragraph'].apply(n_words)
    dataframe['most_frequent_word'] = dataframe['paragraph'].apply(get_most_frequent_word)
    dataframe['most_frequent_word_count'] = dataframe['paragraph'].apply(get_most_frequent_word_count)

    return dataframe

# scraped data is a dataframe that has the 
def task_2_pipeline(data, forum = '', ZSC_labels = None):
    
    # verify structure of data - this line raises Exception if structure not valid
    scraped_data = verify_data_structure(data)
    
    # init the similarity model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    
    # pre-process data - concat Content columns and create paragraphs columns 
    scraped_data = pre_processing_data(scraped_data, min_paragraphs_words = 10)
    scraped_data = words_frequency(scraped_data, col = 'paragraphs')
    
    # disaggregation depends on the forum !
    if forum.lower() == 'wikihow':
        scraped_data = disaggregate_content(scraped_data, tag='div', a_class={"class":"step"})
    elif forum.lower() == 'medium':
        scraped_data = disaggregate_content_medium(scraped_data)
    else:
        scraped_data = disaggregate_content(scraped_data, tag='p')
    
    # create similarty columns : similarity of paragraph to title and Soft skill name
    scraped_data = paragraph_similarity_col(scraped_data, 'Title of URL', model)
    scraped_data = paragraph_similarity_col(scraped_data, 'Soft Skill Name', model)
    
    # Zero-Shot classication - labels : Soft skills
    if ZSC_labels != None:
        scraped_data = zs_classify(scraped_data, ZSC_labels)
        print('shape of data after ZSC : ', scraped_data.shape)
    
    return scraped_data


if __name__ == "__main__": 
    # Read the skills if we want to run ZSC 
    skills = pd.read_excel('../input/softskillsv3/skills_df_v3.xlsx')
    candidate_labels = list(skills['Skills category'].unique())
    
    # read data
    data = pd.read_csv('../input/wikihow-20221029/wiki_how_20221029.csv')
    
    # Run the flow - put ZSC_labels at None if you don't want to run ZSC
    disaggregated_data = task_2_pipeline(data, 'wikihow', ZSC_labels=None )
    
    # Export result
    disaggregated_data.to_csv('disaggregated_data_20221101.csv', index = False)
else: 
    print ("task-2-flow imported !")



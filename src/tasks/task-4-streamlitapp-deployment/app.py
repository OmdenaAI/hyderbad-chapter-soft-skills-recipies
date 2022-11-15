import streamlit as st
import pandas as pd
import templates
import urllib.parse
import utilities
import datetime

def set_session_state():
    # set default values
    if 'search' not in st.session_state:
        st.session_state.search = None
        # print(st.session_state.search)
    if 'tags' not in st.session_state:
        st.session_state.tags = None
    if 'page' not in st.session_state:
        st.session_state.page = 1

    # get parameters in url
    para = st.experimental_get_query_params()
    #st.write(para)
    if 'search' in para:
        st.experimental_set_query_params()
        # decode url
        new_search = urllib.parse.unquote(para['search'][0])
        st.session_state.search = new_search
        # st.write(st.session_state.search)
    if 'tags' in para:
        st.experimental_set_query_params()
        st.session_state.tags = para['tags'][0]
        # st.write(st.session_state.tags)
    if 'page' in para:
        st.experimental_set_query_params()
        st.session_state.page = int(para['page'][0])


if __name__ == '__main__':
    page_size = 5
    set_session_state()
    d=pd.read_csv('Soft-skill-summaries.csv')
    #creating the elastic search client instance
    # es = Elasticsearch('https://gq5ag28mru:hww48y9rk4@coab-testing-434159132.us-east-1.bonsaisearch.net:443')
    st.write(templates.load_css(), unsafe_allow_html=True)
    st.title('Search Soft skills')
    
    if st.session_state.search is None:#if there is no keyword that is searched yet, display the below text input
        search = st.text_input('Enter search words:')
    else: #if there is searched keyword, keep that keyword in the text input box
        search = st.text_input('Enter search words:', st.session_state.search)
    if search:
        if search != st.session_state.search:
            st.session_state.tags = None
        # reset search word
        st.session_state.search = None
        from_i = (st.session_state.page - 1) * page_size
        #search the index 'soft_skills_summaries' for the search word, with the tags of there are any that are activated
        regex_search=utilities.search_words_phrases(search)
        d.scale=datetime.datetime.now()
        result = d[d.Summary.str.contains(regex_search)]
        result.scale=datetime.datetime.now()
        if st.session_state.tags:
          result=result[result['Finer Soft Skill']==st.session_state.tags]
        results=utilities.page_data(result,from_i,page_size,st.session_state.page)
        results.scale=datetime.datetime.now()
        #st.write(results,unsafe_allow_html=True)
        total_hits=len(result)
        # d['Finer Soft Skill']
        # show number of results and time taken
        if total_hits>0:
          st.write(templates.number_of_results(total_hits,(results.scale-d.scale).total_seconds()), unsafe_allow_html=True)
          # # render popular tags as filters
        
          list_of_results=list(result['Finer Soft Skill'].unique())
          if st.session_state.tags is not None and st.session_state.tags not in list_of_results:
                  popular_tags = [st.session_state.tags] + list_of_results
                  # st.write(results['sorted_tags'])
                  # st.write(popular_tags)
                  
                  
          else:
              popular_tags = list_of_results

          popular_tags_html = templates.tag_boxes(search, popular_tags[:10],st.session_state.tags)
          st.markdown(popular_tags_html, unsafe_allow_html=True)
        
    
          # search results
          #st.write(f'''<div style='margin-top: .2rem; margin-bottom: .2rem;'></div>''', unsafe_allow_html=True)
          fig=utilities.word_cloud(results['Summary for word cloud'])
          space_string='      '
          space=f'''
              <div style='margin-top: .15rem; margin-bottom: .15rem;'>
                            {space_string}
              </div><br>
                '''
          st.write(space, unsafe_allow_html=True)
          st.pyplot(fig)
          st.write(space, unsafe_allow_html=True)
          for hit in results['Summary']:
              # result = results['hits']['hits'][i]
              # res = hit['_source']['Summarized Content']
              # res['url'] = result['_id']
              # res['highlights'] = '...'.join(result['highlight']['content'])
              st.write(templates.search_result(hit), unsafe_allow_html=True)
          st.write(space, unsafe_allow_html=True)


          # pagination
          if total_hits >= page_size:
              total_pages = (total_hits + page_size - 1) // page_size
              pagination_html = templates.pagination(total_pages,
                                                      search,
                                                      st.session_state.page,
                                                      st.session_state.tags,)
              st.write(pagination_html, unsafe_allow_html=True)
              
        else:
          st.write(templates.no_result_html(), unsafe_allow_html=True)
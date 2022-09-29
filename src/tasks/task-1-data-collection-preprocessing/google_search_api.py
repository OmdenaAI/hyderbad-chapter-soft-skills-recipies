# -*- coding: utf-8 -*-
"""google-search-api.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RsNDickWzlh68E6YUAapL66W8oHT_k9_

# **Using Google Search API**


1 - create your Programmable Engine here  https://programmablesearchengine.google.com/  → you will get a cx code → save it

2 - Get API KEY here : https://developers.google.com/custom-search/v1/introduction → save it

3 - Use this tow codes within a python code to get search results

# Simple python to get results using this API
"""

import requests
import bs4
import json

cx_code = 'a0fd360ae7bcd4400' # put yours hear (step 1 above)
api_key = 'AIzaSyC7n16Q4QS3yu0GX3iMrN6Zu-nKyQAKpFU'  # put yours hear (step 2 above)
words_to_search = 'soft skills'

url = 'https://customsearch.googleapis.com/customsearch/v1?cx=' + cx_code + '&q=' + words_to_search + '&key=' + api_key

res = requests.get(url)

# res = requests.get('https://customsearch.googleapis.com/customsearch/v1?cx=a0fd360ae7bcd4400&q=soft%20skills&key=AIzaSyC7n16Q4QS3yu0GX3iMrN6Zu-nKyQAKpFU')


results = json.loads(res.text)

for i in range(len(results['items'])):
    print(results['items'][i]['link'])

"""# Ponniah question about What advanced features can be helpful for us?
here https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
we can find a list of parameters to add to the url variable above to custom the research. For example the **lr** parameter sets the language of research : 

"Restricts the search to documents written in a particular language (e.g., lr=lang_ja).

Acceptable values are:

"lang_ar": Arabic

"lang_bg": Bulgarian

"lang_ca": Catalan

"lang_cs": Czech

...."


**the start** parameter : The index of the first result to return. The default number of results per page is 10, so &start=11 would start at the top of the second page of results. Note: The JSON API will never return more than 100 results, even if more than 100 documents match the query, so setting the sum of start + num to a number greater than 100 will produce an error. Also note that the maximum value for num is 10.
"""

# Let's test these two :
url = url + '&' + 'lr=lang_en' + '&' + 'start=11'

res = requests.get(url)

# res = requests.get('https://customsearch.googleapis.com/customsearch/v1?cx=a0fd360ae7bcd4400&q=soft%20skills&key=AIzaSyC7n16Q4QS3yu0GX3iMrN6Zu-nKyQAKpFU')


results = json.loads(res.text)

for i in range(len(results['items'])):
    print(results['items'][i]['link'])



"""You can try the same thing with this parameter :
**filter**  : Controls turning on or off the duplicate content filter.

See Automatic Filtering for more information about Google's search results filters. Note that host crowding filtering applies only to multi-site searches.

By default, Google applies filtering to all search results to improve the quality of those results.

Acceptable values are:

0: Turns off duplicate content filter.

1: Turns on duplicate content filter.
"""

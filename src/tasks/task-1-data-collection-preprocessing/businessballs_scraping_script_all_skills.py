import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime

class businessballs:
    bb = 'https://www.businessballs.com/'

    headers = {
    'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
    'Accept-Langauge' : 'en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,pl;q=0.6,hi;q=0.5',
    'Referer' : bb[:-1],
    'DNT' : '1'
    }

    def scraping_session(self):
        s = requests.Session()
        return s

    def get_article_details(self,articles, skill, criteria,s)->pd.DataFrame:
        art_dict = {key: [] for key in ['Soft Skill Name', 'Criteria', 'URL', 'Title of URL','Content']}
        for article in articles:
            art_dict['Title of URL'].append(article.find('a', class_='col-12').text)
            #art_dict['views'].append(article.select('small')[-1].text)
            link = article.find('a', class_='col-12').get('href')
            art_dict['URL'].append(link)
            body = s.get(link, headers = businessballs.headers)
            so = BeautifulSoup(body.text, 'html.parser')
            t = so.select_one('div.col-12 div.col-12')
            try:
                content = "".join(map(lambda x: str(x), t.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'ul'])))
            except Exception:
                content = "Not scraped"
            art_dict['Content'].append(content)
            #art_dict['metadata'].append("".join(str(so.find_all('meta'))))
            try:
                content = t.text.strip()
                text = re.sub('(\\r)*(\\n)*(\\r)*(\\n).( )*', ' ', content)
                text = re.sub("(\\')?(\\n)*", "", text)
            except AttributeError:
                text = "Not Scraped"
            #art_dict['body'].append(text)
            #art_dict['tracking_log'].append((f'status_code: {body.status_code}', str(datetime.now()), 'Module-businessballs.py', 'Package: BeautifulSoup', 'Dev: Nischay Sabharwal'))
            time.sleep(1)
        art_dict['Soft Skill Name'] = skill
        art_dict['Criteria'] = criteria
        #art_dict['validator_log'] = 'Not Validated'
        return pd.DataFrame(art_dict)

    def category_articles(self, category, mgmt,s)->pd.DataFrame:
        df = pd.DataFrame({})
        for link in mgmt[category]:
            data = s.get(link, headers = businessballs.headers)
            soup = BeautifulSoup(data.text, 'html.parser')
            mva = soup.find_all('div', class_ = 'card-body')
            for i in range(len(mva)):
                try:
                    cardtitle = mva[i].h5.text
                    if cardtitle == 'Most Viewed Resources' or cardtitle == 'Most Popular Resources' or cardtitle == 'Most Liked Resources':
                        articles = mva[i].find_all('div', class_ = 'carousel-item')
                        break 
                except AttributeError:
                    continue
            for ele in articles:
                tag = ele.span.text.strip()
                if  tag != 'Article': 
                    articles.pop(articles.index(ele))
            criteria = (" ".join(re.search(r'\/((\w)+(-\w)*)+\/', link).group()[1:-1].split('-'))).title()
            art = businessballs.get_article_details(self, articles, category, criteria, s)
            df = pd.concat([df, art], ignore_index=True)
            df = df[['Soft Skill Name', 'Criteria', 'URL', 'Title of URL', 'Content']]
            time.sleep(1)
        df.drop_duplicates(inplace = True, ignore_index=True)
        return df

    def scrape_businessballs(self)->pd.DataFrame:
    
        s = businessballs.scraping_session(self)
        data = s.get(businessballs.bb, headers = businessballs.headers)
        soup = BeautifulSoup(data.text, 'html.parser')
        sections = soup.find_all('div', class_ = 'accordion')
        for x in sections:
            button = x.find('div',class_='collapse')
            if button.a.text == 'Management':
                mgmt = button.find_all('div', class_='accordion')
                break
        mgmt = {ele.button.text.strip(): [l.get('href') for l in ele.find_all('a')] for ele in mgmt}
        mgmt.pop('Learn more about the C30')
        mgmt.pop('Finance')

        df = pd.DataFrame({})
        for skill in mgmt.keys():
            temp_df = businessballs.category_articles(self, skill, mgmt, s)
            df = pd.concat([df,temp_df], ignore_index=True)
        df.drop_duplicates(inplace = True, ignore_index=True)
        return df


b = businessballs()
df = b.scrape_businessballs()
df.to_csv('businessballs_data_formatted.csv')
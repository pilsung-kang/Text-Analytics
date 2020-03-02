import requests
import pandas as pd
import re
import time
from bs4 import BeautifulSoup

# url = 'https://arxiv.org/find/all/1/all:+EXACT+text_mining/0/1/0/all/0/1?skip=0'
# response = requests.get(url)
# bs = BeautifulSoup(response.text, 'html.parser')

start = time.clock()
contents = []
for i, skip in enumerate(list(range(0, 175, 25))):

    tmp_url = 'https://arxiv.org/find/all/1/all:+EXACT+text_mining/0/1/0/all/0/1?skip=' + str(skip)
    tmp_response = requests.get(tmp_url)
    tmp_bs = BeautifulSoup(tmp_response.text, 'html.parser')
    tmp_contents = tmp_bs.select_one('div#dlpage').select('span.list-identifier')

    tmp_href = [tmp_href.select_one('a').attrs.get('href') for tmp_href in tmp_contents]
    #tmp_href = list(map(lambda x : x.select_one('a').attrs.get('href'),tmp_contents))
    tmp_list = list(map(lambda x : 'https://arxiv.org' + x, tmp_href))


    for j, tmp_paper in enumerate(tmp_list):
        tmp_response_paper = requests.get(tmp_paper)
        tmp_bs_paper = BeautifulSoup(tmp_response_paper.text, 'html.parser')
        tmp_contents = {'title' : re.sub('\\s+',
                                         ' ',
                                         tmp_bs_paper.select_one('h1.title.mathjax').text.replace('Title:\n', '')),
                         'author' : tmp_bs_paper.select_one('div.authors').text.replace('\n', '').replace('Authors:', ''),
                         'subject' : tmp_bs_paper.select_one('span.primary-subject').text,
                         'abstract' : tmp_bs_paper.select_one('blockquote.abstract.mathjax').text.replace('\nAbstract: ', ''),
                         'meta' : re.sub('\\s+',
                                         ' ',
                                         tmp_bs_paper.select_one('div.submission-history').text.split('[v1]')[1].strip())}
        contents.append(tmp_contents)
        print(j + 1, '/', len(tmp_list))

    print(i + 1, '/', len(list(range(0, 175, 25))))
end = time.clock()
print(end - start)

len(contents)
data = pd.DataFrame(contents)
data.shape
data.iloc[0:2,:]

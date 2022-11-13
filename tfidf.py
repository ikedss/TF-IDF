# Aluno: Leonardo Ikeda

"""
Sua  tarefa  será  gerar  a  matriz  termo-documento  usando  TF-IDF  por  meio  da  aplicação  das 
fórmulas TF-IDF na matriz termo-documento criada com a utilização do algoritmo Bag of Words. Sobre 
o Corpus que recuperamos anteriormente.
"""

from bs4 import BeautifulSoup
import requests
import spacy
import numpy as np
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")


def adiciona_site(site, lsentencas):
    html = requests.get(site)
    soap = BeautifulSoup(html.content, 'html.parser')
    text = soap.get_text()
    token = re.findall('\w+', text)
    pontuacao = ['(', ')', '.', ',', ';', ':', '!', '?','...', '"', '“', '”', '—', '-']

    for palavra in token:
        if palavra not in pontuacao:
            lsentencas.append(palavra.lower())
    return lsentencas


lsentencas = []
sentencas1 = adiciona_site("https://en.wikipedia.org/wiki/Natural_language_processing", lsentencas)
sentencas2 = adiciona_site("https://www.ibm.com/cloud/learn/natural-language-processing", lsentencas)
sentencas3 = adiciona_site("https://www.sas.com/en_us/insights/analytics/what-is-natural-language-processing-nlp.html", lsentencas)
sentencas4 = adiciona_site("https://builtin.com/data-science/high-level-guide-natural-language-processing-techniques", lsentencas)
sentencas5 = adiciona_site("https://deepsense.ai/a-business-guide-to-natural-language-processing-nlp/", lsentencas)


bow = pd.DataFrame(0, index=np.arange(len(lsentencas)), columns=lsentencas)


def bowsum(bow):
    count = 0
    for i in lsentencas:
        bow.at[count, i] += 1
        count += 1
    return bow


newbow = bowsum(bow)

df = pd.DataFrame(data={'sents': lsentencas}, index=[sent for sent in range(len(lsentencas))])

sents = list(map(lambda x: len(x.split(" ")), df['sents']))
tf = newbow.div(sents, axis=0)
idf = np.log(len(newbow)/newbow.sum())
tfidf = tf.multiply(idf, axis=1)
tfidf

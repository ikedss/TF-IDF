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

    for palavra in token:
      ltok.append(palavra)

    sents = []

    sentenca_atual = ""
    for letra in text:
      if letra == "." or letra == "," or letra == "?" or letra == "!" or letra == ";" or letra == "\n" or letra == "\t" or letra == "\t":
          if not sentenca_atual.isspace() and len(sentenca_atual) > 0:
              sents.append(sentenca_atual)
              sentenca_atual = ""
          else:
              sentenca_atual = ""
      else:
          sentenca_atual += letra
    lsentencas.append(sents)

ltok = []
lsentencas = []
sentencas1 = adiciona_site("https://en.wikipedia.org/wiki/Natural_language_processing", lsentencas)
sentencas2 = adiciona_site("https://www.ibm.com/cloud/learn/natural-language-processing", lsentencas)
sentencas3 = adiciona_site("https://www.sas.com/en_us/insights/analytics/what-is-natural-language-processing-nlp.html", lsentencas)
sentencas4 = adiciona_site("https://builtin.com/data-science/high-level-guide-natural-language-processing-techniques", lsentencas)
sentencas5 = adiciona_site("https://deepsense.ai/a-business-guide-to-natural-language-processing-nlp/", lsentencas)

index = []
for i in lsentencas:
  for j in i:
    index.append(j)

bow = pd.DataFrame(0, index=index, columns=ltok)

for i in index[:300]:
  for j in ltok[:300]:
    if j in i:
      bow.loc[i,j] += 1

df = pd.DataFrame(data={'sents': ltok[:300]}, index=[sent for sent in range(len(ltok[:300]))])

sents = list(map(lambda x: len(x.split(" ")), df['sents']))
tf = bow[:300].div(sents, axis=0)
idf = np.log(len(bow)/bow.sum())
tfidf = tf.multiply(idf, axis=1)
tfidf

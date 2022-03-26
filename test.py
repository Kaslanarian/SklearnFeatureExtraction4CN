# from sklearn.feature_extraction.cn_text import CNTfidfVectorizer, CNCountVectorizer
from text import CNTfidfVectorizer, CNCountVectorizer

corpus = [
    "我有一颗星星",
    "你却没有",
    "我在上北京清华大学，北京很美",
]
counter = CNCountVectorizer(stop_words='chinese').fit(corpus)
print(counter.vocabulary_)
print(counter.transform(corpus).todense())
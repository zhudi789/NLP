# coding: utf-8
'''
date: Dec 25.26.2017
describe: 计算英文文本相似性
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 语料
corpus = [
    "This is the first document.",
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
# 查看词频结果
# print(X.toarray())

# 类调用
transformer = TfidfTransformer()
# print transformer
# 将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
# 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
# print(tfidf.toarray())

# 用余弦计算距离
def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)

for vec in tfidf.toarray():
    for vec2 in tfidf.toarray():
        print(cos(vec, vec2))

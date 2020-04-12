# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 语料
corpus = [
    "This is the first document.",
    'This is the second document.',
    'And the third one.',
    'Is this the first document?',
]

# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
# 计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
# 获取词袋中所有文本关键词
vocabulary = vectorizer.get_feature_names()
fp = open("data/stop.csv", 'rb')
list = []
for line in fp.readlines():
    # line = line.replace("\n", "").split(",")
    print(line)
print(list)
final = ""
for word in vocabulary:
    if word not in list:
        final = final+" "+word
print(final)

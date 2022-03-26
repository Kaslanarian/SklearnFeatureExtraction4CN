# sklearn feature-extraction for Chinese:<br>让sklearn的特征提取支持中文

`sklearn`默认是针对英文，而不支持中文分词。即使用中文分词的tokenizer，比如jiaba，也会在其余的地方遇到问题，比如n_gram下的词汇表：

```python
from sklearn.feature_extraction import CountVectorizer
from jieba import lcut

corpus = [
    "我有一颗星星",
    "你却没有",
    "我在上北京清华大学，北京很美",
]

counter = CountVectorizer(
    tokenizer=lcut,
    ngram_range=(1, 2),
).fit(corpus)
print(counter.vocabulary_)
```

输出的词语间是用空格进行间隔，影响可读性，同时如果要对`lcut`指定参数，则要使用`functools.partial`等方法，都不是很方便。因此这里我们基于`jieba`实现了一个可用于处理中文的仿`sklearn`的特征提取接口(`CNCountVectorizer`和`CNTfidfVectorizer`)。

```python
# test.py
from text import CNTfidfVectorizer

corpus = [
    "我有一颗星星",
    "你却没有",
    "我在上北京清华大学，北京很美",
]
counter = CNCountVectorizer(
    ngram_range=(1, 2),
).fit(corpus)
print(counter.vocabulary_)
```

生成的词汇表

```python
{'我': 15, '有': 19, '一颗': 0, '星星': 18, '我有': 17, '有一颗': 20, '一颗星星': 1, '你': 4, '却': 9, '没有': 21, '你却': 5, '却没有': 10, '在': 11, '上': 2, '北京': 6, '清华大学': 22, '，': 25, '很': 13, '美': 24, '我在': 16, '在上': 12, '上北京': 3, '北京清华大学': 8, '清华大学，': 23, '，北京': 26, '北京很': 7, '很美': 14}
```

我们载入了中文停用词，因此可直接指定参数

```python
counter = CNCountVectorizer(
    ngram_range=(1, 2),
    stop_words='chinese',
).fit(corpus)
print(counter.vocabulary_)
```

词汇表变成了

```python
{'我': 7, '有': 9, '一颗': 0, '星星': 8, '你': 2, '却': 4, '没有': 10, '在': 5, '上': 1, '北京': 3, '清华大学': 11, '，': 13, '很': 6, '美': 12}
```

相比于`sklearn`中的特征提取API，我们删除了无用的参数，比如大小写和口音指定，增加了参数HMM，这是`jieba`分词的参数，同时将`analyzer`的可选项修改成了`{'word', 'all', 'search'}`，对应一般分词，全模式分词和搜索模式分词。

## `sklearn`嵌入(Windows)

我们的接口和代码规范借鉴`sklearn`，我们可以将我们的API嵌入`sklearn`。在`Windows`平台上，只要安装了`sklearn`，通过

```bash
python install.py
```

将我们的API经过简单处理后放入`sklearn`包中，它主要做3件事：

1. 将`text.py`重命名为`cn_text.py`放入`sklearn.feature_extraction`文件夹中；
2. 将中文停用词表添加到`sklearn.feature_extraction._stop_words.py`中；
3. 修改`__init__.py`，允许从外部引用我们的API。

这样，我们就可以通过

```python
from sklearn.feature_extraction.cn_text import CNCountVectorizer, TfidfVectorizer
```

在所有路径下使用我们的中文向量化API了。

通过

```bash
python uninstall.py
```

删除我们对`sklearn`进行的修改。

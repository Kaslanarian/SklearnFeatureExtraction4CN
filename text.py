from collections.abc import Mapping
from functools import partial
import numbers
import warnings

import jieba
import numpy as np

from sklearn.base import BaseEstimator
from _stop_words import CHINESE_STOP_WORDS
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, _VectorizerMixin

__all__ = [
    'CHINESE_STOP_WORDS',
    'CNCountVectorizer',
    'CNTfidfVectorizer',
]


def _analyze(doc,
             analyzer=None,
             tokenizer=None,
             ngrams=None,
             decoder=None,
             stop_words=None):
    """Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing or preprocessing.

    If analyzer is used, only the decoder argument is used, as the analyzer is
    intended to replace the preprocessor, tokenizer, and ngrams steps.

    Parameters
    ----------
    analyzer: callable, default=None
    ngrams: callable, default=None
    decoder: callable, default=None
    stop_words: list, default=None

    Returns
    -------
    ngrams: list
        A sequence of tokens, possibly with pairs, triples, etc.
    """
    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc


def _check_stop_list(stop):
    if stop == "chinese":
        return CHINESE_STOP_WORDS
    elif isinstance(stop, str):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:  # assume it's a collection
        return frozenset(stop)


class _CNVectorizerMixin(_VectorizerMixin):
    """Provides common code for text vectorizers (tokenization logic)."""
    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = "".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i:i + n]))

        return tokens

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.

        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        return partial(jieba.lcut, HMM=self.HMM)

    def build_analyzer(self):
        """Return a callable that handles tokenization and n-grams generation.

        Returns
        -------
        analyzer: callable
            A function to handle preprocessing, tokenization
            and n-grams generation.
        """

        if callable(self.analyzer):
            return partial(_analyze,
                           analyzer=self.analyzer,
                           decoder=self.decode)
        if self.analyzer in {'word', 'all', 'search'}:
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return partial(
                _analyze,
                ngrams=self._word_ngrams,
                tokenizer={
                    "word": tokenize,
                    "all": partial(jieba.lcut, cut_all=True, HMM=self.HMM),
                    "search": partial(jieba.lcut_for_search, HMM=self.HMM),
                }[self.analyzer],
                decoder=self.decode,
                stop_words=stop_words,
            )
        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def get_stop_words(self):
        """Build or fetch the effective stop words list.

        Returns
        -------
        stop_words: list or None
                A list of stop words.
        """
        return _check_stop_list(self.stop_words)

    def _validate_params(self):
        """Check validity of ngram_range parameter"""
        min_n, max_m = self.ngram_range
        if min_n > max_m:
            raise ValueError("Invalid value for ngram_range=%s "
                             "lower boundary larger than the upper boundary." %
                             str(self.ngram_range))

    def _warn_for_unused_params(self):
        if (self.ngram_range != (1, 1) and self.ngram_range is not None
                and callable(self.analyzer)):
            warnings.warn("The parameter 'ngram_range' will not be used"
                          " since 'analyzer' is callable'")
        if self.analyzer not in {'word', 'all', 'search'} or callable(
                self.analyzer):
            if self.stop_words is not None:
                warnings.warn(
                    "The parameter 'stop_words' will not be used"
                    " since 'analyzer' not in {'word', 'all', 'search'}")
            if self.tokenizer is not None:
                warnings.warn(
                    "The parameter 'tokenizer' will not be used"
                    " since 'analyzer' not in {'word', 'all', 'search'}")


class CNCountVectorizer(_CNVectorizerMixin, CountVectorizer, BaseEstimator):
    @_deprecate_positional_args
    def __init__(self,
                 *,
                 input='content',
                 encoding='utf-8',
                 decode_error='strict',
                 tokenizer=None,
                 stop_words=None,
                 ngram_range=(1, 1),
                 analyzer='word',
                 HMM=True,
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.HMM = HMM
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral)
                    or max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None" %
                    max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.binary = binary
        self.dtype = dtype


class CNTfidfVectorizer(CNCountVectorizer):
    @_deprecate_positional_args
    def __init__(self,
                 *,
                 input='content',
                 encoding='utf-8',
                 decode_error='strict',
                 tokenizer=None,
                 stop_words=None,
                 ngram_range=(1, 1),
                 analyzer='word',
                 HMM=True,
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.float64,
                 norm='l2',
                 use_idf=True,
                 smooth_idf=True,
                 sublinear_tf=False):

        super().__init__(input=input,
                         encoding=encoding,
                         decode_error=decode_error,
                         tokenizer=tokenizer,
                         analyzer=analyzer,
                         stop_words=stop_words,
                         HMM=HMM,
                         ngram_range=ngram_range,
                         max_df=max_df,
                         min_df=min_df,
                         max_features=max_features,
                         vocabulary=vocabulary,
                         binary=binary,
                         dtype=dtype)

        self._tfidf = TfidfTransformer(norm=norm,
                                       use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning)

    def fit(self, raw_documents, y=None):
        self._check_params()
        self._warn_for_unused_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        check_is_fitted(self, msg='The TF-IDF vectorizer is not fitted')

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

    def _more_tags(self):
        return {'X_types': ['string'], '_skip_test': True}
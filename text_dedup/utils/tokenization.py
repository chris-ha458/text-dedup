#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-12-26 15:59:42
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
from itertools import tee
from typing import List
from typing import Text

import numpy as np


def ngrams(sequence: List[Text], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=1))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=5))
    []
    >>> list(ngrams(["a", "b"], 3, min_length=1))
    [('a', 'b')]
    """
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def ngram_np_chararray(sequence: List[Text], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    np.chararray
        The ngrams packed into a chararray.

    Examples
    --------
    >>> ngram_np_chararray(["a", "b", "c", "d"], 2, min_length=1)
    chararray(['a b', 'b c', 'c d'], dtype='<U3')
    >>> ngram_np_chararray(["a", "b", "c", "d"], 2, min_length=5)
    chararray([''], dtype='<U1')
    >>> ngram_np_chararray(["a", "b"], 3, min_length=1)
    chararray(['a b'], dtype='<U3')
    """
    LEN_SEQUENCE = len(sequence)
    if LEN_SEQUENCE < min_length:
        return np.char.array([""])
    if LEN_SEQUENCE < n:
        return np.char.array(" ".join(sequence))
    return np.char.array([" ".join(sequence[index : index + n]) for index in range(LEN_SEQUENCE - n + 1)])


def ngram_np_vect(sequence: List[Text], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    np.chararray
        The ngrams packed into a chararray.

    Examples
    --------
    >>> ngram_np_vect(["a", "B", "c", "d"], 2, min_length=1)
    array([b'a b', b'b c', b'c d'], dtype='|S3')
    >>> ngram_np_vect(["a", "b", "c", "d"], 2, min_length=5)
    array([b''], dtype='|S1')
    >>> ngram_np_vect(["a", "B"], 3, min_length=1)
    array([b'a b'], dtype='|S3')
    """
    LEN_SEQUENCE = len(sequence)
    if LEN_SEQUENCE < min_length:
        return np.bytes_([""])
    if LEN_SEQUENCE < n:
        return np.char.encode(np.char.lower(np.char.array(" ".join(sequence))))
    return np.char.encode(
        np.char.lower(
            np.array(
                [" ".join(sequence[index : index + n]) for index in range(LEN_SEQUENCE - n + 1)], dtype=np.unicode_
            )
        )
    )

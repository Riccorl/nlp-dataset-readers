from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class Word:
    """
    A word representation that includes text, index in the sentence, POS tag, lemma,
    dependency relation, and similar information.

    # Parameters
    text : `str`, optional
        The text representation.
    index : `int`, optional
        The word offset in the sentence.
    lemma : `str`, optional
        The lemma of this word.
    pos : `str`, optional
        The coarse-grained part of speech of this word.
    dep : `str`, optional
        The dependency relation for this word.
    head : `int`, optional
        The index of the head word.
    """

    text: str
    index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    # preprocessing fields
    lemma: Optional[str] = None
    pos: Optional[str] = None
    dep: Optional[str] = None
    head: Optional[int] = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


class Sentence(list):
    """
    A sentence class, containing a list of `Word`.

    Args:
        words (`List[Word]`):
            The list of words in the sentence.
        id (`int`):
            The sentence id.
    """

    def __init__(self, words: List[Word] = None, id: Any = None):
        super(Sentence, self).__init__()
        self._words = words or []
        self.id = id

    def __len__(self):
        return len(self._words)

    def __getitem__(self, item):
        return self._words[item]

    def __repr__(self):
        return "[" + ", ".join(w.text for w in self._words) + "]"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self._words.__iter__()

    def __next__(self):
        return self._words.__next__()

    def append(self, word: Word):
        self._words.append(word)

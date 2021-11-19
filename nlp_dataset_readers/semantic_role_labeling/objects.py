from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Any, Union

from nlp_dataset_readers.common.objects import Word, Sentence


@dataclass
class Predicate(Word):
    """
    Semantic Role Labeling predicate word. It includes all the fields from `Word` plus
    predicate-related fields from Semantic Role Labeling.

    Args:
        sense (`str`, optional):
            sense of the predicate.
        arguments (`List[Argument]`, optional):
            The list of the arguments of the predicate word.
    """

    sense: Optional[str] = None
    arguments: Optional[List[Argument]] = field(default_factory=list)
    score: float = 0.0

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def add_argument(self, argument: Argument) -> Predicate:
        """
        Adds an argument to the predicate.

        Args:
            argument (`Argument`):
                The argument to add.

        Returns:
            `Predicate`:
                The predicate with the added argument.
        """
        self.arguments.append(argument)
        return self

    @staticmethod
    def from_word(
        word: Word, sense: Optional[str] = None, arguments: Optional[List[Argument]] = None
    ) -> Predicate:
        """
        Create a predicate from a word.

        Args:
            word (`Word`):
                The word to be converted to a predicate.
            sense (`str`, optional):
                The sense of the predicate.
            arguments (`list`, optional):
                The list of arguments of the predicate.

        Returns:
            `Predicate`: The predicate created from the word.
        """
        return Predicate(
            text=word.text,
            index=word.index,
            start_char=word.start_char,
            end_char=word.end_char,
            lemma=word.lemma,
            pos=word.pos,
            dep=word.dep,
            sense=sense,
            arguments=arguments or [],
        )


class Argument:
    """
    Semantic Role Labeling argument span.

    Args:
        role (`str`):
            The label of the argument span.
        predicate (`Predicate`):
            The predicate that has this argument span.
        words (`List[Word]`):
            The list of words that take part in the argument span.
        start_index (`int`):
            The start index of the argument span in the sentence.
        end_index (`int`):
            The end index of the argument span in the sentence.
    """

    def __init__(
        self,
        role: str,
        predicate: Predicate,
        words: List[Word],
        start_index: int,
        end_index: int,
    ):
        self.role: str = role
        self.predicate: Predicate = predicate
        self.words: List[Word] = words
        self.start_index: int = start_index
        self.end_index: int = end_index

    @property
    def span(self):
        return self.start_index, self.end_index

    @property
    def bio_tag(self):
        bio_tags = [f"B-{self.role}"]
        bio_tags += [f"I-{self.role}"] * (self.end_index - self.start_index - 1)
        return bio_tags

    def __str__(self):
        return f"({self.role}, {self.start_index}, {self.end_index})"

    def __repr__(self):
        return self.__str__()


class SrlSentence(Sentence):
    """
    A Semantic Role Labeling Sentence class, used to built the output.

    Args:
        words (`List[Word]`):
            List of `Word` objects.
        id (`Any`):
            The id of the sentence.
    """

    def __init__(self, words: List[Word] = None, id: Any = None):
        super(SrlSentence, self).__init__(words, id)

    def add_predicate(self, predicate: Predicate, index: Optional[int] = None) -> Predicate:
        """
        Add a predicate to the sentence.

        Args:
            predicate (`Predicate`):
                The predicate to add.
            index (`int`, optional):
                The index where to add the predicate.

        Returns:
            `Predicate`: The added predicate.
        """
        if predicate.index is not None and index is None:
            # infer index from predicate
            index = predicate.index
        if index is not None and predicate.index is None:
            # set index of predicate
            predicate.index = index

        if index is None and predicate.index is None:
            raise ValueError("Cannot infer index of predicate")

        if index >= len(self._words):
            raise IndexError(
                f"Index out of range: provided index is {index}, sentence length is {len(self._words)}"
            )
        self._words[index] = predicate
        return predicate

    def get_predicate(self, index: int) -> Predicate:
        """
        Get the predicate at the given index.

        Args:
            index (`int`): The index of the predicate to get.

        Returns:
            `Predicate`: The predicate at the given index.
        """
        if index >= len(self._words):
            raise IndexError(
                f"Index out of range: provided index is {index}, sentence length is {len(self._words)}"
            )
        if not isinstance(self._words[index], Predicate):
            raise TypeError(f"Index {index} is not a predicate")

        return self._words[index]

    @property
    def predicates(self) -> List[Predicate]:
        """
        Get all predicates in the sentence.

        Returns:
            `List[Predicate]`: The list of predicates.
        """
        return [p for p in self._words if isinstance(p, Predicate)]

    def get_predicate_arguments(
        self, predicate: Union[Predicate, int], format: str = "span"
    ) -> Union[List[Argument], List[str]]:
        """
        Get the arguments of the given predicate.

        Args:
            predicate (`Predicate` or `int`):
                The predicate to get the arguments of.
            format (`str`, optional, default: `"span"`):
                The format of the returned arguments, can be either `"span"` or `"bio"`.

        Returns:
            `List[Argument]` or `List[str]`: the arguments of the predicate.
        """
        if isinstance(predicate, Predicate):
            predicate = predicate.index
        arguments = self.get_predicate(predicate).arguments
        if format == "span":
            return arguments
        elif format == "bio":
            bio_tags = ["O"] * len(self)
            for argument in arguments:
                bio_tags[argument.start_index] = f"B-{argument.role}"
                bio_tags[argument.start_index + 1 : argument.end_index] = [f"I-{argument.role}"] * (
                    argument.end_index - argument.start_index - 1
                )
            return bio_tags
        else:
            raise ValueError(f"Unknown format: {format}. Available formats are: `span`, `bio`")

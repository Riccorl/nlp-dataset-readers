from pathlib import Path
from typing import Union, Any, List

import conllu

from nlp_dataset_readers.common.objects import Word
from nlp_dataset_readers.common.reader import Reader
from nlp_dataset_readers.semantic_role_labeling.objects import Argument, Predicate, SrlSentence

from rich.progress import track


class SrlReader(Reader):
    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def parse_sentence(conll_lines: Union[List[str], Any]) -> SrlSentence:
        raise NotImplementedError

    @staticmethod
    def bio_to_spans(bio_tags: List[str], **kwargs) -> List[List[Union[int, str]]]:
        """
        Convert BIO tags to span tags.

        Args:
            bio_tags (`List[str]`):
                BIO tags

        Returns:
            `List[List[Union[int, str]]]`: List of span tags.
        """
        span_tags = []
        for (index, tag) in enumerate(bio_tags):
            # it means no label usually
            if tag in ["O", "_"]:
                continue
            # extract label without BIO prefix
            label = tag[2:]
            # if it is the beginning of a new label, create a new span
            if tag[0] == "B" or len(span_tags) == 0 or label != bio_tags[index - 1][2:]:
                span_tags.append([label, index, -1])
            # close current span
            if (
                index == len(bio_tags) - 1
                or bio_tags[index + 1][0] == "B"
                or label != bio_tags[index + 1][2:]
            ):
                span_tags[-1][2] = index + 1
        return span_tags

    @staticmethod
    def span_to_bio(span_tags: List[List[Union[int, str]]], **kwargs) -> List[str]:
        """
        Convert span tags to BIO tags.

        Args:
            span_tags (`List[List[Union[int, str]]]`):
                Span tags.

        Returns:
            `List[str]`: List of BIO tags.
        """
        raise NotImplementedError

    @staticmethod
    def process_word(word: str) -> str:
        """
        Process word.

        Args:
            word (`str`):
                Word to preprocess.

        Returns:
            `str`: Preprocessed word.
        """
        if word == "-LRB-" or word == "-LSB-":
            return "("
        elif word == "-RRB-" or word == "-RSB-":
            return ")"
        elif word == "-LCB-":
            return "{"
        elif word == "-RCB-":
            return "}"
        elif word == "``" or word == "''":
            return '"'
        elif word == " ":
            # there are parsing error in the data, where " are missing
            # replace empty token with "
            return '"'
        else:
            return word


class Conll2012Reader(SrlReader):
    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        """
        Read a CoNLL-2012 file and return a list of SRL sentences.

        Args:
            file_path (`Union[str, Path]`):
                Path to the CoNLL-2012 file
            *args:
                Positional arguments
            **kwargs:
                Keyword arguments

        Returns:
            `List[SrlSentence]`: List of SRL sentences
        """
        parsed_sentences = []
        file_path = Path(file_path)
        # CoNLL-2012 files are usually split into multiple files
        if file_path.is_dir():
            files = list(file_path.glob("**/*.gold_conll"))
        # but it can also be a single file
        else:
            files = [file_path]
        # read all files
        for file_name in track(files, description="Reading files"):
            parsed_sentences.extend(Conll2012Reader.read_file(file_name))
        return parsed_sentences

    @staticmethod
    def read_file(file_name: Union[str, Path]) -> List[SrlSentence]:
        """
        Read a single CoNLL-2012 file and return a list of SRL sentences.

        Args:
            file_name (`Union[str, Path]`):
                Path to the CoNLL-2012 file

        Returns:
            `List[SrlSentence]`: List of SRL sentences
        """
        # output data structure
        parsed_sentences = []
        with open(file_name) as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line:
                    sentence.append(line)
                else:
                    parsed_sentences.append(Conll2012Reader.parse_sentence(sentence))
                    sentence = []
        return parsed_sentences

    @staticmethod
    def parse_sentence(conll_lines: List[str]) -> SrlSentence:
        """
        Parse a single CoNLL-2012 sentence and return a SRL sentence.

        Args:
            conll_lines (`List[str]`):
                List of CoNLL-2012 lines.

        Returns:
            `SrlSentence`: Parsed sentence with predicates and arguments.
        """
        # take the first line for some preliminary information
        conll_line = conll_lines[0].split()
        # sentence id is not unique
        sentence = SrlSentence()
        # empty lists to collect the SRL BIO labels
        span_labels = [[] for _ in conll_line[11:-1]]
        # Create variables representing the current label for each label
        # sequence we are collecting.
        current_span_labels = [None for _ in conll_line[11:-1]]
        _conll_lines = [line.split() for line in conll_lines]
        # read the sentence lines and build the sentence structure
        # with Words, Predicates and Arguments
        for line in conll_lines:
            conll_components = line.split()
            word = Word(
                text=Conll2012Reader.process_word(conll_components[3]),
                index=int(conll_components[2]),
                lemma=conll_components[6] if conll_components[6] != "-" else None,
                pos=conll_components[4],
            )
            # if line[6] is not "-", then the word is a predicate
            if conll_components[7] != "-":
                # if PropBank, the sense is line[6].line[7]
                if conll_components[6].isdigit():
                    sense = f"{conll_components[6]}.{conll_components[7]}"
                # otherwise, the sense is line[7]
                else:
                    sense = conll_components[7]
                word = Predicate.from_word(word, sense)
            sentence.append(word)

            # read into BIO labels
            for annotation_index, annotation in enumerate(conll_components[11:-1]):
                # strip all bracketing information to
                # get the actual propbank label.
                label = annotation.strip("()*")
                if "(" in annotation:
                    # Entering into a span for a particular semantic role label.
                    # We append the label and set the current span for this annotation.
                    bio_label = "B-" + label
                    span_labels[annotation_index].append(bio_label)
                    current_span_labels[annotation_index] = label
                elif current_span_labels[annotation_index] is not None:
                    # If there's no '(' token, but the current_span_label is not None,
                    # then we are inside a span.
                    bio_label = "I-" + current_span_labels[annotation_index]
                    span_labels[annotation_index].append(bio_label)
                else:
                    # We're outside a span.
                    span_labels[annotation_index].append("O")
                # Exiting a span, so we reset the current span label for this annotation.
                if ")" in annotation:
                    current_span_labels[annotation_index] = None

        # Now we can parse the arguments
        argument_spans = [Conll2012Reader.bio_to_spans(labels) for labels in span_labels]
        for predicate_index, argument_span in enumerate(argument_spans):
            for role_name, start, end in argument_span:
                if role_name == "V":
                    continue
                argument = Argument(
                    role=role_name,
                    predicate=sentence.predicates[predicate_index],
                    words=sentence[start:end],
                    start_index=start,
                    end_index=end,
                )
                sentence.predicates[predicate_index].add_argument(argument)
        return sentence


class Conll2009Reader(SrlReader):
    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        """
        Read a CoNLL-2009 file and return a list of SRL sentences.

        Args:
            file_path (`Union[str, Path]`):
                Path to the CoNLL-2009 file
            *args:
                Positional arguments
            **kwargs:
                Keyword arguments

        Returns:
            `List[SrlSentence]`: List of SRL sentences
        """
        # output data structure
        sentences = []
        with open(file_path) as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line:
                    sentence.append(line)
                else:
                    sentences.append(Conll2009Reader.parse_sentence(sentence))
                    sentence = []
        return sentences

    @staticmethod
    def parse_sentence(conll_lines: List[str]) -> SrlSentence:
        """
        Parse a CoNLL-2009 sentence and return a SRL sentence.

        Args:
            conll_lines (`List[str]`):
                List of CoNLL-2009 lines.

        Returns:
            `SrlSentence`: Parsed sentence with predicates and arguments.
        """
        # conll 2009 doesn't have a sentence id
        sentence = SrlSentence()
        for line in conll_lines:
            line = line.split()
            word = Word(
                text=Conll2009Reader.process_word(line[1]),  # line[1] is the word
                index=int(line[0]) - 1,  # line[0] is the index, it starts at 1
                lemma=line[2],  # line[2] is the gold lemma
                pos=line[4],  # line[4] is the gold POS tag
                # line[10] is dependency label, if provided
                dep=line[10] if len(line) > 10 and line[10] != "_" else None,
            )
            # head is a particular case, it needs
            # to be handled separately
            if line[8] == "_":  # no head
                head = None
            elif line[8] == "0":  # root
                head = 0
            else:
                head = int(line[8]) - 1  # line[8] is the gold head
            word.head = head
            # if line[12] is "Y", then the word is a predicate
            if len(line) > 12 and line[12] == "Y":
                word = Predicate.from_word(word, line[13])
            sentence.append(word)

        # parse arguments
        for line in conll_lines:
            line = line.split()
            word_index = int(line[0]) - 1
            for i, arg in enumerate(line[14:]):
                if arg != "_":
                    predicate = sentence.predicates[i]
                    predicate.add_argument(
                        Argument(
                            arg,
                            predicate,
                            sentence[word_index : word_index + 1],
                            start_index=word_index,
                            end_index=word_index + 1,
                        )
                    )
        return sentence


class UnitedSrlReader(SrlReader):
    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        """
        Read a CoNLL-U file and return a list of SRL sentences.

        Args:
            file_path (`Union[str, Path]`): Path to the CoNLL-2009 file
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            `List[SrlSentence]`: List of SRL sentences
        """
        # output data structure
        sentences = []
        with open(file_path, "r", encoding="utf-8") as conll_file:
            for sentence in conllu.parse_incr(
                conll_file,
                fields=["id", "form", "lemma", "frame", "roles"],
                field_parsers={"roles": lambda line, i: line[i:]},
            ):
                sentences.append(sentence)
        sentences = [
            UnitedSrlReader.parse_sentence(sentence)
            for sentence in track(sentences, description="Reading file")
        ]
        return sentences

    @staticmethod
    def parse_sentence(conll_lines: conllu.TokenList) -> SrlSentence:
        """
        Parse a CoNLL-U sentence and return a SRL sentence.

        Args:
            conll_lines (`conllu.TokenList`):
                List of CoNLL-U lines in conllu.TokenList format.

        Returns:
            `SrlSentence`: Parsed sentence with predicates and arguments.
        """
        # final sentence id is a combination of document id and sentence id
        sentence = SrlSentence(
            id=f"{conll_lines.metadata['document_id']}_{conll_lines.metadata['sentence_id']}"
        )
        # Add words and predicates
        for token in conll_lines:
            word = Word(
                UnitedSrlReader.process_word(token["form"]), token["id"], lemma=token["lemma"]
            )
            if token["frame"] != "_":
                word = Predicate.from_word(word, token["frame"])
            sentence.append(word)
        # No arguments, early return
        if any("roles" not in token for token in conll_lines):
            return sentence
        # Add arguments
        roles_list = []
        for predicate_index in range(len(conll_lines[0]["roles"])):
            roles = [conll_lines[i]["roles"][predicate_index] for i in range(len(conll_lines))]
            # infer if it is dependency or span based dataset
            # if role != "B-V" because in dep data verbs are still B-V
            is_span = any(role.startswith("B-") for role in roles if role != "B-V")
            if is_span:
                # convert argument to span
                roles = UnitedSrlReader.bio_to_spans(roles)
            else:
                # threat dependency as span of length 1
                roles = [(role, i, i + 1) for i, role in enumerate(roles) if role != "_"]
            roles_list.append(roles)
        # add arguments
        for predicate_index, argument_span in enumerate(roles_list):
            for role_name, start, end in argument_span:
                if role_name in ["B-V", "V"]:
                    continue
                try:
                    argument = Argument(
                        role=role_name,
                        predicate=sentence.predicates[predicate_index],
                        words=sentence[start:end],
                        start_index=start,
                        end_index=end,
                    )
                except:
                    # TODO : fix this in the dataset, it should not happen
                    continue
                sentence.predicates[predicate_index].add_argument(argument)
        return sentence

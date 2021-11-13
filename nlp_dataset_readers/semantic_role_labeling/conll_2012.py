from pathlib import Path
from typing import Union, Any, List

from common.objects import Word
from semantic_role_labeling.objects import SrlSentence, Predicate, Argument
from semantic_role_labeling.srl_reader import SrlReader


class Conll2012Reader(SrlReader):
    def read(self, file_path: Union[str, Path], *args, **kwargs) -> Any:
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
            for file_name in file_path.glob("**/*.gold_conll"):
                parsed_sentences.extend(self.read_file(file_name))
        # but it can also be a single file
        else:
            parsed_sentences.extend(self.read_file(file_path))
        return parsed_sentences

    def read_file(self, file_name: Union[str, Path]) -> List[SrlSentence]:
        # output data structure
        parsed_sentences = []
        with open(file_name) as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line == "":
                    parsed_sentences.append(self.parse_sentence(sentence))
                    sentence = []
                else:
                    sentence.append(line)
        return parsed_sentences

    def parse_sentence(self, conll_lines: List[str]) -> SrlSentence:
        # take the first line for some preliminary information
        conll_line = conll_lines[0].split("\t")
        # get the sentence id
        sentence = SrlSentence(id=conll_line[1])
        # empty lists to collect the SRL BIO labels
        span_labels = [[] for _ in conll_line[11:-1]]
        # Create variables representing the current label for each label
        # sequence we are collecting.
        current_span_labels = [None for _ in conll_line[11:-1]]
        _conll_lines = [line.split("\t") for line in conll_lines]
        # read the sentence lines and build the sentence structure
        # with Words, Predicates and Arguments
        for line in conll_lines:
            conll_components = line.split("\t")
            word = Word(
                text=conll_components[3],
                index=int(conll_components[2]),
                lemma=conll_components[6] if conll_lines[6] != "-" else None,
                pos=conll_components[4],
            )
            # if line[6] is not "-", then the word is a predicate
            if conll_components[7] != "-":
                # if propbank, the sense is line[6].line[7]
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
        argument_spans = [self.bio_to_spans(labels) for labels in span_labels]
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

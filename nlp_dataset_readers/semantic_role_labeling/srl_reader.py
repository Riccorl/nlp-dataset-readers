from pathlib import Path
from typing import Union, Any, List, Tuple

from common.reader import Reader


class SrlReader(Reader):
    def read(self, file_path: Union[str, Path], *args, **kwargs) -> Any:
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

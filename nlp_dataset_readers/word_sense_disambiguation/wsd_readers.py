from pathlib import Path
from typing import Any, Union

from nlp_dataset_readers.common.reader import Reader


class WsdReader(Reader):
    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        raise NotImplementedError

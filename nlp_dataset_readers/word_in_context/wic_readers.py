from pathlib import Path
from typing import Any, Union

from nlp_dataset_readers.common.reader import Reader


class WicReader(Reader):
    def read(self, file_path: Union[str, Path], *args, **kwargs) -> Any:
        raise NotImplementedError

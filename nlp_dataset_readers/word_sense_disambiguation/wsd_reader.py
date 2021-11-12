from pathlib import Path
from typing import Any, Union


class Reader:
    def __init__(self):
        pass

    def read(self, file_path: Union[str, Path]) -> Any:
        raise NotImplementedError

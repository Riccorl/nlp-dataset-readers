from pathlib import Path
from typing import Any, Union


class Reader:
    def __init__(self):
        pass

    @staticmethod
    def read(file_path: Union[str, Path], *args, **kwargs) -> Any:
        raise NotImplementedError

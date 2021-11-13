from pathlib import Path
from typing import Any, Union

from common.reader import Reader


class WsdReader(Reader):
    def read(self, file_path: Union[str, Path], *args, **kwargs) -> Any:
        raise NotImplementedError

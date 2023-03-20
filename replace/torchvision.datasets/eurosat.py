import os
from typing import Callable, Optional

from .folder import ImageFolder
from .utils import download_and_extract_archive


class EuroSAT(ImageFolder):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """
    idx_to_class = {
        0: 'annual crop land',
        1: 'a forest',
        2: 'brushland or shrubland',
        3: 'a highway or a road',
        4: 'industrial buildings',
        5: 'pasture land',
        6: 'permanent crop land',
        7: 'residential buildings',
        8: 'a river',
        9: 'a sea or a lake'
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template = "A centered satellite photo of {}."
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(self._data_folder, transform=transform, target_transform=target_transform)
        self.root = os.path.expanduser(root)

        self.prompt_template = prompt_template
        self.clip_prompts = [ 
            prompt_template.format(label.lower().replace('_', ' ').replace('-', ' ')) \
            for label in self.idx_to_class.values()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self._base_folder, exist_ok=True)
        download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root=self._base_folder,
            md5="c8fa014336c82ac7804f0398fcb19387",
        )

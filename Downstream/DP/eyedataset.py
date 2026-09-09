import re
from pathlib import Path
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class EyeSubjectDataset(Dataset):
    """
    Expected directory structure:

        root/
        ├── class_0/
        │   ├── 024910-20210107@135523-L5-S.png
        │   └── ...
        └── class_1/
            ├── 031245-20220318@101530-R2-S.png
            └── ...

    Each item returns:

        image, label, subject_id, path

    subject_id is defined as:

        person_id + "_" + eye_side

    Example:

        024910-20210107@135523-L5-S.png
        -> subject_id = "024910_L"
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    FILENAME_PATTERN = re.compile(
        r"^(?P<person_id>[^-]+)"
        r"-.*"
        r"-(?P<eye_side>[LR])"
        r"(?P<image_index>\d+)"
        r"(?:-[^.]+)?"
        r"\.[^.]+$",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        root_dir,
        transform=None,
        strict_filename=True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.strict_filename = strict_filename

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root_dir}"
            )

        class_dirs = sorted(
            directory
            for directory in self.root_dir.iterdir()
            if directory.is_dir()
        )

        if not class_dirs:
            raise RuntimeError(
                f"No class directories found under {self.root_dir}"
            )

        # class_0 -> 0, class_1 -> 1
        self.class_to_idx = {}

        for class_dir in class_dirs:
            match = re.fullmatch(
                r"class_(\d+)",
                class_dir.name,
            )

            if match is None:
                raise ValueError(
                    "Class directory names must follow the pattern "
                    f"'class_<number>', but got: {class_dir.name}"
                )

            self.class_to_idx[class_dir.name] = int(
                match.group(1)
            )

        self.samples = []

        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]

            for image_path in sorted(class_dir.rglob("*")):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower()
                    not in self.IMAGE_EXTENSIONS
                ):
                    continue

                try:
                    person_id, eye_side, image_index = (
                        self.parse_filename(image_path.name)
                    )
                except ValueError:
                    if self.strict_filename:
                        raise

                    print(
                        "[EyeSubjectDataset] Skip unrecognized file: "
                        f"{image_path}"
                    )
                    continue

                subject_id = f"{person_id}_{eye_side}"

                self.samples.append(
                    {
                        "path": image_path,
                        "label": label,
                        "person_id": person_id,
                        "eye_side": eye_side,
                        "image_index": image_index,
                        "subject_id": subject_id,
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"No valid image files found under {self.root_dir}"
            )

        print(
            f"[EyeSubjectDataset] Loaded {len(self.samples)} images, "
            f"{len(self.get_subject_ids())} subjects, "
            f"{len(self.class_to_idx)} classes."
        )

    @classmethod
    def parse_filename(
        cls,
        filename: str,
    ) -> Tuple[str, str, int]:
        """
        Example:

            024910-20210107@135523-L5-S.png

        Returns:

            ("024910", "L", 5)
        """
        match = cls.FILENAME_PATTERN.match(filename)

        if match is None:
            raise ValueError(
                "Filename does not match the expected pattern: "
                f"{filename}"
            )

        person_id = match.group("person_id")
        eye_side = match.group("eye_side").upper()
        image_index = int(match.group("image_index"))

        return person_id, eye_side, image_index

    def get_subject_ids(self):
        return sorted(
            {
                sample["subject_id"]
                for sample in self.samples
            }
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = sample["path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = sample["label"]
        subject_id = sample["subject_id"]

        return (
            image,
            label,
            subject_id,
            str(image_path),
        )



class ImageListDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        list_file,
        transform=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        with open(
            list_file,
            "r",
            encoding="utf-8",
        ) as file_handle:
            self.relative_paths = [
                line.strip()
                for line in file_handle
                if line.strip()
            ]

        if len(self.relative_paths) == 0:
            print(
                f"[Warning] No images in list: {list_file}"
            )

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, index):
        relative_path = Path(
            self.relative_paths[index]
        )

        # image_path = (
        #     self.dataset_root
        #     / relative_path
        # )

        # image = Image.open(
        #     image_path
        # ).convert("RGB")


        image_path = (
            self.dataset_root
            / relative_path
        )

        # Some recorded paths may omit the extension.
        # Try ".jpg" if the original path does not exist.
        if not image_path.exists():
            jpg_path = image_path.with_suffix(".jpg")

            if jpg_path.exists():
                image_path = jpg_path
            else:
                raise FileNotFoundError(
                    f"Image not found:\n"
                    f"  {image_path}\n"
                    f"Also tried:\n"
                    f"  {jpg_path}"
                )

        image = Image.open(
            image_path
        ).convert("RGB")

        class_name = relative_path.parts[0]

        if class_name == "class_0":
            label = 0
        elif class_name == "class_1":
            label = 1
        else:
            raise ValueError(
                f"Unknown class directory: {class_name}"
            )

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            label,
            str(image_path),
        )


if __name__ == "__main__":
    print("Testing EyeSubjectDataset...")

    # 只测试完全未见病人的图像：
    # test_dataset = ImageListDataset(
    #     dataset_root=(
    #         "/path/to/workspace/dataset/"
    #         "DeepUWF/SH_DR/shrdr/test"
    #     ),
    #     list_file=(
    #         "/path/to/workspace/dataset/"
    #         "DeepUWF/SH_DR/shrdr/test_overlap_lists/"
    #         "subject_0_eye_0_sample_0.txt"
    #     ),
    #     transform=test_transform,
    # )

    # 只测试“同病人但不同眼睛”的图像：
    # list_file = (
    #     "/path/to/workspace/dataset/"
    #     "DeepUWF/SH_DR/shrdr/test_overlap_lists/"
    #     "subject_1_eye_0_sample_0.txt"
    # )

    #只测试“同病人且同一只眼、但不同样本”的图像：
#     list_file = (
#     "/path/to/workspace/dataset/"
#     "DeepUWF/SH_DR/shrdr/test_overlap_lists/"
#     "subject_1_eye_1_sample_0.txt"
# )
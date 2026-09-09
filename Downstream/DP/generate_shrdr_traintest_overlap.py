import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

#本代码中的subject是指patient,而不是dp_client中的subject=patient+eye_side
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Example:
# 000084-20181016@145240-L2-S.jpg
#
# patient_id = 000084
# eye_side   = L
# sample stem = complete filename without extension
SHDR_FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)"
    r"-(?P<date>\d{8})"
    r"@(?P<time>\d{6})"
    r"-(?P<eye_side>[LR])"
    r"(?P<image_index>\d+)"
    r"-S$",
    re.IGNORECASE,
)



def parse_shdr_filename(image_path):
    """
    Parse an SHDR image filename.

    Example:
        000084-20181016@145240-L2-S.jpg

    Returns:
        {
            "patient_id": "000084",
            "eye_side": "L",
            "eye_id": "000084_L",
            "sample_id": "000084-20181016@145240-L2-S",
            ...
        }
    """
    image_path = Path(image_path)
    stem = image_path.stem

    match = SHDR_FILENAME_PATTERN.match(stem)

    if match is None:
        raise ValueError(
            f"Unexpected SHDR filename format: {image_path.name}"
        )

    values = match.groupdict()

    patient_id = values["patient_id"]
    eye_side = values["eye_side"].upper()

    return {
        "patient_id": patient_id,
        "eye_side": eye_side,
        "eye_id": f"{patient_id}_{eye_side}",
        "sample_id": stem,
        "date": values["date"],
        "time": values["time"],
        "image_index": int(values["image_index"]),
    }


def collect_shdr_records(root_dir):
    """
    Scan an ImageFolder-style directory:

        root_dir/
            class_0/
            class_1/

    Returns one record per image.
    """
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root_dir}"
        )

    records = []
    invalid_files = []

    for image_path in sorted(root_dir.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            parsed = parse_shdr_filename(image_path)
        except ValueError:
            invalid_files.append(str(image_path))
            continue

        relative_path = image_path.relative_to(root_dir)

        # Expected first-level directory: class_0 or class_1
        class_name = (
            relative_path.parts[0]
            if len(relative_path.parts) >= 2
            else "unknown"
        )

        record = {
            **parsed,
            "class_name": class_name,
            "filename": image_path.name,
            "absolute_path": str(image_path.resolve()),
            "relative_path": str(relative_path),
        }

        records.append(record)

    if invalid_files:
        print(
            f"[Warning] {len(invalid_files)} files could not be parsed."
        )

        for path in invalid_files[:10]:
            print(f"  {path}")

    return records


def classify_test_overlap(
    test_record,
    train_patient_ids,
    train_eye_ids,
    train_sample_ids,
):
    """
    Assign one test image to an overlap category.

    Returns:
        subject_overlap, eye_overlap, sample_overlap, category_name
    """
    subject_overlap = int(
        test_record["patient_id"] in train_patient_ids
    )

    eye_overlap = int(
        test_record["eye_id"] in train_eye_ids
    )

    sample_overlap = int(
        test_record["sample_id"] in train_sample_ids
    )

    # Logical consistency checks
    if sample_overlap and not eye_overlap:
        raise RuntimeError(
            "A sample overlap was found without an eye overlap: "
            f"{test_record['filename']}"
        )

    if eye_overlap and not subject_overlap:
        raise RuntimeError(
            "An eye overlap was found without a subject overlap: "
            f"{test_record['filename']}"
        )

    category_name = (
        f"subject_{subject_overlap}_"
        f"eye_{eye_overlap}_"
        f"sample_{sample_overlap}"
    )

    return (
        subject_overlap,
        eye_overlap,
        sample_overlap,
        category_name,
    )


def write_txt(records, output_path, path_mode="relative"):
    """
    Write one image per line.

    path_mode:
        "filename": only the filename
        "relative": class_0/xxx.jpg
        "absolute": full absolute path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_map = {
        "filename": "filename",
        "relative": "relative_path",
        "absolute": "absolute_path",
    }

    if path_mode not in field_map:
        raise ValueError(
            f"Unsupported path_mode: {path_mode}"
        )

    output_field = field_map[path_mode]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        for record in sorted(
            records,
            key=lambda item: item["relative_path"],
        ):
            file_handle.write(
                record[output_field] + "\n"
            )


def generate_overlap_test_lists(
    train_root,
    test_root,
    output_dir,
    path_mode="relative",
):
    """
    Categorize all images under test_root according to their overlap
    with train_root.

    Generated categories:
        subject_0_eye_0_sample_0
        subject_1_eye_0_sample_0
        subject_1_eye_1_sample_0
        subject_1_eye_1_sample_1

    A separate file is also generated for each class.
    """
    train_root = Path(train_root)
    test_root = Path(test_root)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_records = collect_shdr_records(train_root)
    test_records = collect_shdr_records(test_root)

    train_patient_ids = {
        record["patient_id"]
        for record in train_records
    }

    train_eye_ids = {
        record["eye_id"]
        for record in train_records
    }

    train_sample_ids = {
        record["sample_id"]
        for record in train_records
    }

    groups = defaultdict(list)

    for record in test_records:
        (
            subject_overlap,
            eye_overlap,
            sample_overlap,
            category_name,
        ) = classify_test_overlap(
            test_record=record,
            train_patient_ids=train_patient_ids,
            train_eye_ids=train_eye_ids,
            train_sample_ids=train_sample_ids,
        )

        classified_record = {
            **record,
            "subject_overlap": subject_overlap,
            "eye_overlap": eye_overlap,
            "sample_overlap": sample_overlap,
            "category": category_name,
        }

        groups[category_name].append(
            classified_record
        )

    # All logically valid categories are created, even when empty.
    valid_categories = [
        "subject_0_eye_0_sample_0",
        "subject_1_eye_0_sample_0",
        "subject_1_eye_1_sample_0",
        "subject_1_eye_1_sample_1",
    ]

    # Write TXT lists.
    for category_name in valid_categories:
        category_records = groups.get(
            category_name,
            [],
        )

        # Complete category list
        write_txt(
            records=category_records,
            output_path=(
                output_dir
                / f"{category_name}.txt"
            ),
            path_mode=path_mode,
        )

        # Separate lists for class_0 and class_1
        for class_name in ["class_0", "class_1"]:
            class_records = [
                record
                for record in category_records
                if record["class_name"] == class_name
            ]

            write_txt(
                records=class_records,
                output_path=(
                    output_dir
                    / f"{category_name}_{class_name}.txt"
                ),
                path_mode=path_mode,
            )

    # Also write a detailed CSV manifest.
    manifest_path = output_dir / "overlap_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        fieldnames = [
            "category",
            "subject_overlap",
            "eye_overlap",
            "sample_overlap",
            "class_name",
            "patient_id",
            "eye_side",
            "eye_id",
            "date",
            "time",
            "image_index",
            "filename",
            "relative_path",
            "absolute_path",
        ]

        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for category_name in valid_categories:
            for record in sorted(
                groups.get(category_name, []),
                key=lambda item: item["relative_path"],
            ):
                writer.writerow(
                    {
                        field_name: record[field_name]
                        for field_name in fieldnames
                    }
                )

    # Write summary CSV.
    summary_path = output_dir / "overlap_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.writer(file_handle)

        writer.writerow(
            [
                "category",
                "subject_overlap",
                "eye_overlap",
                "sample_overlap",
                "num_total",
                "num_class_0",
                "num_class_1",
                "num_patients",
                "num_eyes",
            ]
        )

        for category_name in valid_categories:
            records = groups.get(
                category_name,
                [],
            )

            class_counts = Counter(
                record["class_name"]
                for record in records
            )

            if records:
                first = records[0]
                subject_overlap = first["subject_overlap"]
                eye_overlap = first["eye_overlap"]
                sample_overlap = first["sample_overlap"]
            else:
                match = re.match(
                    r"subject_(\d)_eye_(\d)_sample_(\d)",
                    category_name,
                )

                subject_overlap = int(match.group(1))
                eye_overlap = int(match.group(2))
                sample_overlap = int(match.group(3))

            writer.writerow(
                [
                    category_name,
                    subject_overlap,
                    eye_overlap,
                    sample_overlap,
                    len(records),
                    class_counts.get("class_0", 0),
                    class_counts.get("class_1", 0),
                    len({
                        record["patient_id"]
                        for record in records
                    }),
                    len({
                        record["eye_id"]
                        for record in records
                    }),
                ]
            )

    print("=" * 80)
    print("SHDR train/test overlap categorization")
    print("=" * 80)

    print(f"Train images: {len(train_records)}")
    print(f"Test images:  {len(test_records)}")

    assigned_total = 0

    for category_name in valid_categories:
        records = groups.get(
            category_name,
            [],
        )

        assigned_total += len(records)

        class_counts = Counter(
            record["class_name"]
            for record in records
        )

        print(
            f"{category_name}: "
            f"total={len(records)}, "
            f"class_0={class_counts.get('class_0', 0)}, "
            f"class_1={class_counts.get('class_1', 0)}, "
            f"patients={len(set(r['patient_id'] for r in records))}, "
            f"eyes={len(set(r['eye_id'] for r in records))}"
        )

    print("-" * 80)
    print(
        f"Assigned test images: "
        f"{assigned_total}/{len(test_records)}"
    )

    if assigned_total != len(test_records):
        raise RuntimeError(
            "Some test images were not assigned exactly once."
        )

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Detailed manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Split SHDR test images into train/test overlap categories."
        )
    )

    parser.add_argument(
        "--train_root",
        type=str,
        default=(
            "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train"
        ),
    )

    parser.add_argument(
        "--test_root",
        type=str,
        default=(
            "/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/test"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/shrdr_test_overlap_lists",
    )

    parser.add_argument(
        "--path_mode",
        type=str,
        choices=[
            "filename",
            "relative",
            "absolute",
        ],
        default="relative",
        help=(
            "How paths are written to TXT files. "
            "'relative' writes class_0/xxx.jpg."
        ),
    )

    args = parser.parse_args()

    generate_overlap_test_lists(
        train_root=args.train_root,
        test_root=args.test_root,
        output_dir=args.output_dir,
        path_mode=args.path_mode,
    )
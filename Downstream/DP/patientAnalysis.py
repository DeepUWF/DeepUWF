#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_uwf_filename(path):
    """
    Parse UWF filename.

    Expected example:
        024910-20210107@135523-L5-S.png

    Returns:
        patient_id
        patient_eye_id
        eye
        visit_date

    Notes
    -----
    For the example above:

        patient_id     = 024910
        eye            = L
        patient_eye_id = 024910_L
        visit_date     = 20210107
    """

    stem = Path(path).stem

    # ---------------------------------------------------------
    # Patient ID
    #
    # Everything before the first "-"
    # ---------------------------------------------------------
    patient_id = stem.split("-")[0]

    # ---------------------------------------------------------
    # Visit date
    #
    # Expected:
    # 024910-20210107@135523-L5-S
    #        ^^^^^^^^
    # ---------------------------------------------------------
    visit_match = re.search(
        r"-(\d{8})@",
        stem,
    )

    if visit_match is not None:
        visit_date = visit_match.group(1)
    else:
        visit_date = "UNKNOWN"

    # ---------------------------------------------------------
    # Eye
    #
    # Expected eye field such as:
    # -L5-
    # -R3-
    #
    # We only care about L / R.
    # ---------------------------------------------------------
    eye_match = re.search(
        r"-(L|R)\d*(?:-|$)",
        stem,
        flags=re.IGNORECASE,
    )

    if eye_match is not None:
        eye = eye_match.group(1).upper()
    else:
        eye = "UNKNOWN"

    if eye != "UNKNOWN":
        patient_eye_id = (
            f"{patient_id}_{eye}"
        )
    else:
        patient_eye_id = (
            f"{patient_id}_UNKNOWN"
        )

    return {
        "patient_id": patient_id,
        "patient_eye_id": patient_eye_id,
        "eye": eye,
        "visit_date": visit_date,
    }


def percentile(values, q):
    """
    Simple percentile without NumPy dependency.
    """

    if len(values) == 0:
        return float("nan")

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    position = (
        (len(values) - 1)
        * q
        / 100.0
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = position - lower

    return (
        values[lower]
        * (1.0 - fraction)
        + values[upper]
        * fraction
    )


def print_distribution(
    title,
    values,
):
    if not values:
        print(
            f"\n{title}: no data"
        )
        return

    print(
        f"\n{'=' * 70}"
    )
    print(title)
    print(
        f"{'=' * 70}"
    )

    print(
        f"Count:   {len(values)}"
    )
    print(
        f"Mean:    {mean(values):.3f}"
    )
    print(
        f"Median:  {median(values):.3f}"
    )
    print(
        f"Min:     {min(values)}"
    )
    print(
        f"25%:     {percentile(values, 25):.3f}"
    )
    print(
        f"75%:     {percentile(values, 75):.3f}"
    )
    print(
        f"90%:     {percentile(values, 90):.3f}"
    )
    print(
        f"95%:     {percentile(values, 95):.3f}"
    )
    print(
        f"Max:     {max(values)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze patient-level statistics "
            "for the SH_DR/UWF population."
        )
    )

    parser.add_argument(
        "--data_root",
        type=str,
        #required=True,
        default="/path/to/workspace/dataset/DeepUWF/SH_DR/shrdr/train",
        help=(
            "Root directory containing the "
            "training images."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./patient_statistics",
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root does not exist: "
            f"{data_root}"
        )

    # =========================================================
    # Find all images
    # =========================================================

    image_paths = sorted(
        [
            path
            for path in data_root.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found under "
            f"{data_root}"
        )

    print(
        f"Found {len(image_paths)} images."
    )

    # =========================================================
    # Data structures
    # =========================================================

    patient_to_images = defaultdict(
        list
    )

    patient_eye_to_images = defaultdict(
        list
    )

    patient_to_eyes = defaultdict(
        set
    )

    patient_to_visits = defaultdict(
        set
    )

    patient_to_labels = defaultdict(
        set
    )

    patient_eye_to_visits = defaultdict(
        set
    )

    eye_counter = Counter()

    label_counter = Counter()

    records = []

    unknown_eye_count = 0
    unknown_visit_count = 0

    # =========================================================
    # Parse images
    # =========================================================

    for image_path in image_paths:

        parsed = parse_uwf_filename(
            image_path
        )

        patient_id = (
            parsed["patient_id"]
        )

        patient_eye_id = (
            parsed["patient_eye_id"]
        )

        eye = parsed["eye"]

        visit_date = (
            parsed["visit_date"]
        )

        # -----------------------------------------------------
        # Assume immediate parent directory is class label.
        # -----------------------------------------------------
        label = (
            image_path.parent.name
        )

        patient_to_images[
            patient_id
        ].append(
            str(image_path)
        )

        patient_eye_to_images[
            patient_eye_id
        ].append(
            str(image_path)
        )

        patient_to_labels[
            patient_id
        ].add(
            label
        )

        label_counter[
            label
        ] += 1

        if eye != "UNKNOWN":
            patient_to_eyes[
                patient_id
            ].add(
                eye
            )

            eye_counter[
                eye
            ] += 1
        else:
            unknown_eye_count += 1

        if visit_date != "UNKNOWN":
            patient_to_visits[
                patient_id
            ].add(
                visit_date
            )

            patient_eye_to_visits[
                patient_eye_id
            ].add(
                visit_date
            )
        else:
            unknown_visit_count += 1

        records.append(
            {
                "path": str(
                    image_path
                ),
                "filename": (
                    image_path.name
                ),
                "label": label,
                "patient_id": (
                    patient_id
                ),
                "patient_eye_id": (
                    patient_eye_id
                ),
                "eye": eye,
                "visit_date": (
                    visit_date
                ),
            }
        )

    # =========================================================
    # Basic statistics
    # =========================================================

    num_images = len(
        image_paths
    )

    num_patients = len(
        patient_to_images
    )

    num_patient_eyes = len(
        patient_eye_to_images
    )

    images_per_patient = [
        len(images)
        for images
        in patient_to_images.values()
    ]

    images_per_patient_eye = [
        len(images)
        for images
        in patient_eye_to_images.values()
    ]

    visits_per_patient = [
        len(
            patient_to_visits[
                patient_id
            ]
        )
        for patient_id
        in patient_to_images
    ]

    visits_per_patient_eye = [
        len(
            patient_eye_to_visits[
                patient_eye_id
            ]
        )
        for patient_eye_id
        in patient_eye_to_images
    ]

    # =========================================================
    # Patient thresholds
    # =========================================================

    patient_thresholds = [
        1,
        2,
        3,
        5,
        10,
        20,
    ]

    # =========================================================
    # Eye composition
    # =========================================================

    patients_left_only = 0
    patients_right_only = 0
    patients_both_eyes = 0
    patients_unknown_eye = 0

    for patient_id in (
        patient_to_images
    ):
        eyes = patient_to_eyes[
            patient_id
        ]

        if eyes == {"L"}:
            patients_left_only += 1

        elif eyes == {"R"}:
            patients_right_only += 1

        elif (
            "L" in eyes
            and "R" in eyes
        ):
            patients_both_eyes += 1

        else:
            patients_unknown_eye += 1

    # =========================================================
    # Label consistency per patient
    # =========================================================

    mixed_label_patients = {
        patient_id: labels
        for patient_id, labels
        in patient_to_labels.items()
        if len(labels) > 1
    }

    # =========================================================
    # Print summary
    # =========================================================

    print(
        "\n"
        + "=" * 70
    )
    print(
        "SH_DR / UWF PATIENT-LEVEL SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"Data root:           "
        f"{data_root}"
    )

    print(
        f"Total images:        "
        f"{num_images}"
    )

    print(
        f"Total patients:      "
        f"{num_patients}"
    )

    print(
        f"Total patient-eyes:  "
        f"{num_patient_eyes}"
    )

    print(
        f"Unknown eye images:  "
        f"{unknown_eye_count}"
    )

    print(
        f"Unknown visit images:"
        f"  {unknown_visit_count}"
    )

    # ---------------------------------------------------------
    # Label distribution
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )
    print(
        "IMAGE CLASS DISTRIBUTION"
    )
    print(
        "=" * 70
    )

    for label, count in sorted(
        label_counter.items()
    ):
        percentage = (
            100.0
            * count
            / num_images
        )

        print(
            f"{label:<25} "
            f"{count:>8} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Images / patient
    # ---------------------------------------------------------

    print_distribution(
        "IMAGES PER PATIENT",
        images_per_patient,
    )

    print(
        "\nPatient counts by "
        "minimum number of images:"
    )

    for threshold in (
        patient_thresholds
    ):
        count = sum(
            value >= threshold
            for value
            in images_per_patient
        )

        percentage = (
            100.0
            * count
            / num_patients
        )

        print(
            f"  >= {threshold:2d} images: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Images / patient-eye
    # ---------------------------------------------------------

    print_distribution(
        "IMAGES PER PATIENT-EYE",
        images_per_patient_eye,
    )

    # ---------------------------------------------------------
    # Visits
    # ---------------------------------------------------------

    print_distribution(
        "VISITS PER PATIENT",
        visits_per_patient,
    )

    print_distribution(
        "VISITS PER PATIENT-EYE",
        visits_per_patient_eye,
    )

    # ---------------------------------------------------------
    # Eye statistics
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )
    print(
        "EYE DISTRIBUTION"
    )
    print(
        "=" * 70
    )

    print(
        f"Left-eye images:     "
        f"{eye_counter.get('L', 0)}"
    )

    print(
        f"Right-eye images:    "
        f"{eye_counter.get('R', 0)}"
    )

    print(
        f"Unknown-eye images:  "
        f"{unknown_eye_count}"
    )

    print(
        "\nPatient eye composition:"
    )

    print(
        f"Left only:           "
        f"{patients_left_only}"
    )

    print(
        f"Right only:          "
        f"{patients_right_only}"
    )

    print(
        f"Both eyes:           "
        f"{patients_both_eyes}"
    )

    print(
        f"Unknown/other:       "
        f"{patients_unknown_eye}"
    )

    # ---------------------------------------------------------
    # Multi-visit patients
    # ---------------------------------------------------------

    num_multi_visit = sum(
        value >= 2
        for value
        in visits_per_patient
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "LONGITUDINAL INFORMATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Patients with >=2 visits: "
        f"{num_multi_visit} / "
        f"{num_patients} "
        f"({100.0 * num_multi_visit / num_patients:.2f}%)"
    )

    # ---------------------------------------------------------
    # Patient-label consistency
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )
    print(
        "PATIENT LABEL CONSISTENCY"
    )
    print(
        "=" * 70
    )

    print(
        f"Patients appearing in "
        f"multiple class folders: "
        f"{len(mixed_label_patients)}"
    )

    if mixed_label_patients:
        print(
            "\nWARNING:"
        )

        print(
            "Some patients appear in more than one "
            "class. This must be handled carefully "
            "when constructing patient-level "
            "train/validation/test splits."
        )

        for patient_id, labels in list(
            mixed_label_patients.items()
        )[:20]:
            print(
                f"  {patient_id}: "
                f"{sorted(labels)}"
            )

    # =========================================================
    # Save image-level CSV
    # =========================================================

    image_csv_path = (
        output_dir
        / "image_level_metadata.csv"
    )

    with image_csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "filename",
                "label",
                "patient_id",
                "patient_eye_id",
                "eye",
                "visit_date",
            ],
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    # =========================================================
    # Save patient-level CSV
    # =========================================================

    patient_csv_path = (
        output_dir
        / "patient_level_statistics.csv"
    )

    with patient_csv_path.open(
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "patient_id",
            "num_images",
            "num_eyes",
            "eyes",
            "num_visits",
            "labels",
            "num_labels",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for patient_id in sorted(
            patient_to_images
        ):

            writer.writerow(
                {
                    "patient_id": (
                        patient_id
                    ),
                    "num_images": len(
                        patient_to_images[
                            patient_id
                        ]
                    ),
                    "num_eyes": len(
                        patient_to_eyes[
                            patient_id
                        ]
                    ),
                    "eyes": ",".join(
                        sorted(
                            patient_to_eyes[
                                patient_id
                            ]
                        )
                    ),
                    "num_visits": len(
                        patient_to_visits[
                            patient_id
                        ]
                    ),
                    "labels": ",".join(
                        sorted(
                            patient_to_labels[
                                patient_id
                            ]
                        )
                    ),
                    "num_labels": len(
                        patient_to_labels[
                            patient_id
                        ]
                    ),
                }
            )

    # =========================================================
    # Save patient-eye CSV
    # =========================================================

    patient_eye_csv_path = (
        output_dir
        / "patient_eye_statistics.csv"
    )

    with patient_eye_csv_path.open(
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "patient_eye_id",
            "num_images",
            "num_visits",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for patient_eye_id in sorted(
            patient_eye_to_images
        ):

            writer.writerow(
                {
                    "patient_eye_id": (
                        patient_eye_id
                    ),
                    "num_images": len(
                        patient_eye_to_images[
                            patient_eye_id
                        ]
                    ),
                    "num_visits": len(
                        patient_eye_to_visits[
                            patient_eye_id
                        ]
                    ),
                }
            )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "OUTPUT FILES"
    )
    print(
        "=" * 70
    )

    print(
        f"Image metadata:      "
        f"{image_csv_path}"
    )

    print(
        f"Patient statistics:  "
        f"{patient_csv_path}"
    )

    print(
        f"Patient-eye stats:   "
        f"{patient_eye_csv_path}"
    )


if __name__ == "__main__":
    main()
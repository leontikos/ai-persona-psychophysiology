#!/usr/bin/env python
"""Arrange the files of the data archive (RepOD, https://doi.org/10.18150/IPGR44) under
$STUDY_DATA_ROOT in the layout the analysis scripts expect (README.md, "Data layout").

Download the dataset's files into one folder, choosing "Original File Format" for the survey
table, and run

    python analysis/utils/prepare_repod_data.py path/to/downloads

The script
  - unpacks every .zip archive in the folder into $STUDY_DATA_ROOT (conversation_logs/,
    psychophysiology_signal/, CONVERSATION_ANALYSES/);
  - writes SURVEY_data/AI_boss_qualtrics_quality.csv from AI_boss_data_pretest_posttest, the
    semicolon-separated survey and expert-rating table. In the archive the three expert raters'
    columns are labelled Judge_A, Judge_B and Judge_C; the scripts call them Judge1-3, in the same
    order (A = Judge 1, B = Judge 2, C = Judge 3), and expect each rater's mean rating in a column
    named Average_JudgeN.

The tab-separated version that the repository generates for display (.tab) alters line breaks in
the free-text columns (the task deliverables and the open-ended feedback), which the deliverable-
provenance and open-ended analyses read, so the script stops if only that version is present.
"""
import glob
import os
import sys
import zipfile

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
JUDGES = {"A": "Judge1", "B": "Judge2", "C": "Judge3"}


def survey_table(src: str) -> str:
    """Path of the survey table in its original (semicolon-separated) format."""
    hits = [p for p in glob.glob(os.path.join(src, "AI_boss_data_pretest_posttest*"))
            if not p.endswith(".tab")]
    if hits:
        return hits[0]
    if glob.glob(os.path.join(src, "AI_boss_data_pretest_posttest*.tab")):
        sys.exit("Found only the .tab version of AI_boss_data_pretest_posttest. Download it again "
                 "with 'Original File Format'; the .tab version alters line breaks in the free-text "
                 "columns.")
    sys.exit(f"AI_boss_data_pretest_posttest not found in {src}")


def rename_judges(columns) -> dict:
    ren = {}
    for c in columns:
        for letter, judge in JUDGES.items():
            if c == f"Judge_{letter}":
                ren[c] = f"Average_{judge}"
            elif c.endswith(f"_Judge_{letter}"):
                ren[c] = c[: -len(f"_Judge_{letter}")] + f"_{judge}"
    return ren


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    os.makedirs(ROOT, exist_ok=True)

    for z in sorted(glob.glob(os.path.join(src, "*.zip"))):
        with zipfile.ZipFile(z) as fh:
            fh.extractall(ROOT)
        print(f"unpacked {os.path.basename(z)}")

    path = survey_table(src)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=rename_judges(df.columns))
    out = os.path.join(ROOT, "SURVEY_data", "AI_boss_qualtrics_quality.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} participants, {df.shape[1]} columns)")

    for d in ("conversation_logs", "psychophysiology_signal", "CONVERSATION_ANALYSES/successful_analyses"):
        if not os.path.isdir(os.path.join(ROOT, d)):
            print(f"note: {d}/ not found under {ROOT}; the steps that read it will not run")


if __name__ == "__main__":
    main()

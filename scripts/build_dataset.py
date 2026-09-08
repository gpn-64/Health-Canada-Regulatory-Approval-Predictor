"""Raw workbook -> data/processed/submissions_clean.csv, with a QA summary.

Run from the project root:

    python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import CleaningReport, load_completed  # noqa: E402


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report = CleaningReport()
    df = load_completed(report)
    df.to_csv(C.CLEAN_FILE, index=False)

    print(f"Wrote {C.CLEAN_FILE.relative_to(C.PROJECT_ROOT)}  ({len(df):,} rows)")
    print("\nQA summary")
    print("-" * 60)
    print(json.dumps(report.as_dict(), indent=2))

    print("\nSanity checks")
    print("-" * 60)
    print(f"missing accepted dates : {df['date_accepted'].isna().sum()}")
    print(f"missing concluded dates: {df['date_concluded'].isna().sum()}")
    print(f"negative review_days   : {(df['review_days'] < 0).sum()}")
    print(f"review_days  min/median/max: "
          f"{df['review_days'].min()} / {df['review_days'].median():.0f} / {df['review_days'].max()}")
    print(f"approved rate          : {df['approved'].mean():.3f}")
    print(f"submission_type counts : {df['submission_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

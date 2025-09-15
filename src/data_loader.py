from __future__ import annotations
import os
import pandas as pd

DEFAULT_CSV = 'data/sample_marks.csv'

def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    path = csv_path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Dataset not found at {path}. Make sure the CSV exists (see data/sample_marks.csv).'
        )
    df = pd.read_csv(path)
    required = {'study_hours', 'assign_avg', 'attendance', 'final_mark', 'passed'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {missing}')
    return df

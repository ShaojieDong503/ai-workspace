"""Write the DuckDB SQL for the full-data profile of the prediction table.

Independent of the pipeline: column list read from the parquet schema, every
aggregate computed by DuckDB over every row (no sampling).
"""
import pyarrow.parquet as pq
from pathlib import Path
ROOT = Path(r"D:\ai-workspace\project")
cols = pq.read_schema(ROOT / "data/processed/prediction/2015/prediction_2015_07.parquet").names
num = [c for c in cols if c not in ("cell_id", "date")]
agg = []
for c in num:
    f = f'CAST("{c}" AS DOUBLE)'
    agg += [f'sum(CASE WHEN "{c}" IS NULL OR isnan({f}) THEN 1 ELSE 0 END) AS "{c}__nan"',
            f'sum(CASE WHEN {f} = 0 THEN 1 ELSE 0 END) AS "{c}__zero"',
            f'sum(CASE WHEN {f} < 0 THEN 1 ELSE 0 END) AS "{c}__neg"',
            f'min(CASE WHEN isnan({f}) THEN NULL ELSE {f} END) AS "{c}__min"',
            f'max(CASE WHEN isnan({f}) THEN NULL ELSE {f} END) AS "{c}__max"']
sql = f"""
SET threads=10; SET memory_limit='9GB'; SET preserve_insertion_order=false;
COPY (
SELECT regexp_extract(filename, 'prediction_(\d{{4}}_\d{{2}})', 1) AS ym,
       count(*) AS n_rows,
       count(DISTINCT cell_id) AS n_cells,
       count(DISTINCT CAST(date AS DATE)) AS n_days,
       min(date) AS dmin, max(date) AS dmax,
       {', '.join(agg)}
FROM read_parquet('data/processed/prediction/*/prediction_*.parquet', filename=true)
GROUP BY ym ORDER BY ym
) TO 'audit/evidence/pred_profile.csv' (HEADER);
COPY (
SELECT regexp_extract(filename, 'prediction_(\d{{4}}_\d{{2}})', 1) AS ym,
       count(*) - count(DISTINCT (cell_id, date)) AS dup_keys
FROM read_parquet('data/processed/prediction/*/prediction_*.parquet', filename=true)
GROUP BY ym ORDER BY ym
) TO 'audit/evidence/pred_dupkeys.csv' (HEADER);
"""
(ROOT / "audit/scripts/a01_profile.sql").write_text(sql, encoding="utf-8")
print(len(num), "columns")

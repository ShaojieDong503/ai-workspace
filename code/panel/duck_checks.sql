-- DuckDB checks for the cell-day panel.
-- Run with:
--   $env:PYTHONPATH='D:\ai-workspace\project\.pylibs'
--   C:\Python314\python.exe -c "import duckdb; duckdb.sql(open(r'code/panel/duck_checks.sql').read()).show()"
-- or paste individual queries into `python -m duckdb`-style sessions.

-- 1. rows per month file
-- expect 163787 x days_in_month
select regexp_extract(filename, 'prediction_(\d{4}_\d{2})', 1) as month,
       count(*) as rows
from read_parquet('D:\ai-workspace\project\data\processed\prediction\*\*.parquet')
group by all
order by month;

-- 2. month completeness: rows == days in month x 163787
select regexp_extract(filename, 'prediction_(\d{4}_\d{2})', 1) as month,
       count(*) as rows,
       cast(date_part('day', last_day(date)) as int) as days,
       count(*) = 163787 * cast(date_part('day', last_day(date)) as int) as complete
from read_parquet('D:\ai-workspace\project\data\processed\prediction\*\*.parquet')
group by filename, days
order by month;

-- 3. duplicate (cell_id, date) keys: expect 0
select count(*) as duplicate_keys from (
  select cell_id, date, count(*) c
  from read_parquet('D:\ai-workspace\project\data\processed\prediction\*\*.parquet')
  group by 1, 2 having c > 1);

-- 4. schema and dtypes
describe select * from read_parquet(
  'D:\ai-workspace\project\data\processed\prediction\2015\prediction_2015_07.parquet');

-- 5. null counts per column, one month (the gap-free contract)
select count(*) as rows,
       count(*) - count(aod055)          as aod055_nan,
       count(*) - count(aod047)          as aod047_nan,
       count(*) - count(hms_any)         as hms_nan,
       count(*) - count(merra_pblh)      as pblh_nan,
       count(*) - count(fire_count_100km) as fire_nan
from read_parquet('D:\ai-workspace\project\data\processed\prediction\2015\prediction_2015_07.parquet');

-- 6. date coverage: first and last day per file, and the day count
select regexp_extract(filename, '(\d{4}_\d{2})', 1) as month,
       min(date) as first_day, max(date) as last_day,
       count(distinct date) as distinct_days
from read_parquet('D:\ai-workspace\project\data\processed\prediction\*/*.parquet')
group by all
order by month;

-- 7. training table profile
select count(*) as rows, count(distinct cell_id) as cells,
       min(date) as first_day, max(date) as last_day,
       round(min(pm25), 2) as pm25_min, round(max(pm25), 2) as pm25_max,
       count(*) filter (where pm25 < 0) as pm25_negative,
       count(*) filter (where aod055 is null) as no_aod
from read_parquet('D:\ai-workspace\project\data\interim\scratch\target2015\target_table.parquet');

-- 8. training vs prediction: rows present in one but not the other (expect 0)
with t as (select cell_id, date from read_parquet(
             'D:\ai-workspace\project\data\interim\scratch\target2015\target_table.parquet')
           where date >= '2015-07-01' and date < '2015-08-01'),
     p as (select cell_id, date from read_parquet(
             'D:\ai-workspace\project\data\processed\prediction\2015\prediction_2015_07.parquet'))
select (select count(*) from (select * from t except select * from p)) as in_training_only,
       (select count(*) from (select * from p except select * from t)) as in_prediction_only;
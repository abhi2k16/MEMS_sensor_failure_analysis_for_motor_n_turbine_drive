# MEMS Sensor Failure Analysis — Motor & Turbine Drive

Analysis of lateral vibration (`VIB_LAT`), lateral shock (`SHK_LAT`), and nitrogen flow rate (`N2_RATE`) from MEMS sensors across failed and intact coiled tubing drilling jobs. Jobs are categorised by drive type: **motor** or **turbine**.

---

## Repository Structure

```
memsFailJobsMotorTurbine/
├── config.py          # All constants: directories, job IDs, categories, activities, columns
├── data_loader.py     # CSV loading, cleaning, partitioning, activity extraction
├── snv_analysis.py    # Shock & vibration analysis and data aggregation functions
├── plotting.py        # All visualisation functions (bar, scatter, boxplot, histogram)
├── main.py            # Single executable entry point — runs the full pipeline
└── triel_code.py      # Original monolithic reference script
```

---

## Requirements

### Python Packages
```
pandas
numpy
matplotlib
ctd_partitioner
```

Install via pip:
```bash
pip install pandas numpy matplotlib
```

> `ctd_partitioner` is an internal package. Ensure it is available on the Python path before running.

---

## Data Setup

Before running, place your CSV files in the following directories (configured in `config.py`):

| Job Type    | Directory Path                                                   |
|-------------|------------------------------------------------------------------|
| Failed Jobs | `*/Data/AllJobsData`           |
| Intact Jobs | `*/DataWOFail/AllJobsData`     |

### Expected Filename Format
```
O.<id>_concat_data.csv
```
Example: `O.1048592.41-7_concat_data.csv`

### Required Columns in Each CSV

| Column      | Description                    |
|-------------|--------------------------------|
| `DateTime`  | Timestamp (parsed to `TIME`)   |
| `BVEL`      | Bit velocity                   |
| `CT_WGT`    | Coiled tubing weight           |
| `DEPT`      | Depth                          |
| `HDTH`      | Hole depth                     |
| `FLWI`      | Flow rate in                   |
| `APRS_RAW`  | Annulus pressure (raw)         |
| `IPRS_RAW`  | Internal pressure (raw)        |
| `N2_RATE`   | Nitrogen flow rate             |
| `VIB_LAT`   | Lateral vibration [g]          |
| `SHK_LAT`   | Lateral shock [g]              |

---

## Configuration (`config.py`)

All key settings are centralised here. Modify this file to adapt the analysis to new datasets.

| Variable               | Description                                              |
|------------------------|----------------------------------------------------------|
| `directory_failed_jobs`| Path to failed job CSV files                             |
| `directory_intact_jobs`| Path to intact job CSV files                             |
| `job_ids_motor_turbine`| Dict mapping job IDs to `motor` or `turbine` drive type  |
| `job_categories`       | `["failed_job_csv", "intact_job_csv"]`                   |
| `activities`           | List of 5 drilling activities to analyse                 |
| `columns_of_interest`  | List of columns extracted from each CSV                  |

### Motor Jobs (8)
```
O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x9, O.xxxxxxxx.xx-1x,
O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x,O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x6, O.xxxxxxxx.xx-1x
```

### Turbine Jobs (9)
```
O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x9, O.xxxxxxxx.xx-1x,
O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x,O.xxxxxxxx.xx-1x, O.xxxxxxxx.xx-1x6, O.xxxxxxxx.xx-1x
```

---

## How to Run the Analysis

### Step 1 — Clone the repository
```bash
git clone https://github.com/abhi2k16/MEMS_sensor_failure_analysis_for_motor_n_turbine_drive.git
cd MEMS_sensor_failure_analysis_for_motor_n_turbine_drive
```

### Step 2 — Set up data directories
Update the paths in `config.py` if your data is stored in a different location:
```python
directory_failed_jobs = "your/path/to/failed/jobs"
directory_intact_jobs = "your/path/to/intact/jobs"
```

### Step 3 — Add job IDs
If analysing new jobs, add their IDs to the `job_ids_motor_turbine` dict in `config.py`:
```python
job_ids_motor_turbine = {
    'motor': [..., 'O.XXXXXXX.XX-X'],
    'turbine': [..., 'O.XXXXXXX.XX-X']
}
```

### Step 4 — Run the pipeline
```bash
python main.py
```

---

## Pipeline Execution Order (`main.py`)

The pipeline runs these steps sequentially:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `load_csv_files()` | Globs all CSV files from failed and intact directories |
| 2 | `extract_and_label_job_ids()` | Parses job IDs from filenames, appends `[M]` or `[T]` label |
| 3 | `clean_data()` | Parses `DateTime` → `TIME`, selects columns, coerces to numeric |
| 4 | `partition_all_jobs()` | Runs `ctd_partitioner` on each job; skips jobs that fail |
| 5 | `get_jobs_activities()` | Extracts unique activity list per job from partition output |
| 6 | `build_activity_dicts()` | Extracts VIB/SHK arrays per job per activity |
| 7 | `aggregate_activity_data()` | Combines VIB/SHK across all jobs per activity |
| 8 | `aggregate_whole_run_data()` | Concatenates all activity data into whole-run arrays |
| 9 | All `plot_*()` functions | Generates all visualisations |

---

## Module Reference

### `data_loader.py`

| Function | Description |
|----------|-------------|
| `load_csv_files()` | Returns dict of CSV file paths for failed and intact jobs |
| `extract_and_label_job_ids(csv_files_allJobs)` | Extracts job IDs via regex, labels with `[M]`/`[T]`, loads CSVs |
| `clean_data(job_ids, job_id_wise_csv_data)` | Converts datetime, selects columns, coerces all values to numeric |
| `partition_all_jobs(job_ids, job_wise_cleaned_data)` | Partitions each job by activity; returns updated job ID list excluding failures |
| `get_jobs_activities(updated_job_ids, job_wise_partitioned_data)` | Returns unique activities per job |

### `snv_analysis.py`

| Function | Description |
|----------|-------------|
| `mems_SnV_analysis_alljobs(activities, partitioner_data, main_data)` | Extracts VIB/SHK per activity, masks outliers (VIB > 100g, SHK > 600g), returns arrays and level counts |
| `mean_median_max_values(parameter_data, activities, job_ids)` | Computes mean, median, max per activity for failed jobs only |
| `build_activity_dicts(activities, updated_job_ids, ...)` | Builds nested dicts: `{job_type: {activity: {job_id: [data]}}}` |
| `aggregate_activity_data(activities, updated_job_ids, ...)` | Flattens per-job data into per-activity lists across all jobs |
| `aggregate_whole_run_data(activities, updated_job_ids, ...)` | Concatenates all activities into single failed/intact arrays |

### `plotting.py`

| Function | Plot Type | Description |
|----------|-----------|-------------|
| `plot_mean_median_max_bars()` | Bar chart | Mean, median, max VIB & SHK per failed job per activity |
| `plot_activity_boxplots()` | Box plot | VIB & SHK distribution per activity — failed vs. intact |
| `plot_shock_histogram_bar()` | Bar chart | SHK count and % distribution across 0–600g bins (50g steps) |
| `plot_shock_histogram_combined()` | Histogram + box | SHK histogram (log scale) with boxplot overlay |
| `plot_vib_histogram_bar()` | Bar chart | VIB count and % distribution across 0–70g bins (10g steps) |
| `plot_vib_histogram_combined()` | Histogram + box | VIB histogram (log scale) with boxplot overlay |
| `plot_vib_level_count_bars()` | Bar chart | VIB count above thresholds: 10g, 12.5g, 15g per job |
| `plot_shk_level_count_bars()` | Bar chart | SHK count above threshold: 500g per job |
| `plot_vib_scatter_motor_turbine()` | Scatter | VIB counts by range split by motor/turbine and failed/intact |
| `plot_shock_scatter_motor_turbine()` | Scatter | SHK counts by range split by motor/turbine and failed/intact |
| `plot_n2_vib_shk_boxplots()` | Box plot | N2 flow rate and SHK distribution per job |

---

## Activities Analysed

These five activities cover the majority of a drill run:

| Activity | Description |
|----------|-------------|
| `On Bottom Drilling` | Active drilling at bottom of hole |
| `Pull Test` | Tension test on coiled tubing |
| `Wiper Trip` | Cleaning pass through the wellbore |
| `Trip In Run` | Running coiled tubing into the well |
| `Trip Out Run` | Pulling coiled tubing out of the well |

---

## Outlier Thresholds

| Parameter | Valid Range | Action on Violation |
|-----------|-------------|---------------------|
| `VIB_LAT` | 0 – 100 g   | Masked (set to NaN) |
| `SHK_LAT` | 0 – 600 g   | Masked (set to NaN) |
| `N2_RATE` | ≥ 0         | Filtered out        |

---

## Applications

- **Failure Analysis** — Identify VIB/SHK patterns that differentiate failed from intact jobs
- **Drive Type Comparison** — Compare motor vs. turbine behaviour across shock and vibration ranges
- **Activity Monitoring** — Understand sensor behaviour during specific drilling activities
- **Predictive Maintenance** — Flag jobs with high VIB/SHK counts as potential failure candidates

# MEMS Shock & Vibration Analysis — Failed vs. Intact Jobs (Motor & Turbine)

Analysis of lateral vibration and shock data from MEMS sensors across failed and intact drilling jobs, categorised by drive type (motor/turbine).

---

## Repository Structure

```
memsFailJobsMotorTurbine/
├── config.py                              # Constants: directories, job IDs, categories, activities, columns
├── data_loader.py                         # CSV loading, cleaning, partitioning, activity extraction
├── snv_analysis.py                        # S&V analysis functions and data aggregation
├── plotting.py                            # All plotting functions
├── main.py                                # Executable entry point
└── memsWnWOFailJobsIncludeMotorTurbine.py # Original monolithic script (reference)
```

---

## Data

Two sets of CSV files are expected, each containing concatenated time-series sensor data per job:

| Job Type     | Directory                                                        |
|--------------|------------------------------------------------------------------|
| Failed Jobs  | `D:/2025/DataAnalysisShockMEMSSensor/Data/AllJobsData`           |
| Intact Jobs  | `D:/2025/DataAnalysisShockMEMSSensor/DataWOFail/AllJobsData`     |

Expected filename format: `O.<id>_concat_data.csv`

### Columns Used

| Column     | Description                  |
|------------|------------------------------|
| `TIME`     | Timestamp (from `DateTime`)  |
| `BVEL`     | Bit velocity                 |
| `CT_WGT`   | Coiled tubing weight         |
| `DEPT`     | Depth                        |
| `HDTH`     | Hole depth                   |
| `FLWI`     | Flow rate in                 |
| `APRS_RAW` | Annulus pressure (raw)       |
| `IPRS_RAW` | Internal pressure (raw)      |
| `N2_RATE`  | Nitrogen flow rate           |
| `VIB_LAT`  | Lateral vibration [g]        |
| `SHK_LAT`  | Lateral shock [g]            |

---

## Job Classification

Jobs are labelled with `[M]` (motor) or `[T]` (turbine) based on predefined job ID lists in `config.py`.

---

## Activities Analysed

- On Bottom Drilling
- Pull Test
- Wiper Trip
- Trip In Run
- Trip Out Run

Partitioning is performed using `ctd_partitioner`.

---

## Modules

### `config.py`
Centralised configuration — data directories, job ID motor/turbine mapping, job categories, activity list, and column names.

### `data_loader.py`

| Function                    | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `load_csv_files()`          | Globs CSV files from failed and intact job directories   |
| `extract_and_label_job_ids()` | Parses job IDs from filenames and appends `[M]`/`[T]` label |
| `clean_data()`              | Parses datetime, selects columns, coerces to numeric     |
| `partition_all_jobs()`      | Runs `ctd_partitioner` on each job, handles failures     |
| `get_jobs_activities()`     | Extracts unique activities per job from partition data   |

### `snv_analysis.py`

| Function                        | Description                                                        |
|---------------------------------|--------------------------------------------------------------------|
| `mems_SnV_analysis_alljobs()`   | Extracts VIB/SHK data per activity, applies outlier masking        |
| `mean_median_max_values()`      | Computes mean, median, max per activity per failed job             |
| `build_activity_dicts()`        | Builds per-job, per-activity VIB/SHK data dictionaries            |
| `aggregate_activity_data()`     | Aggregates VIB/SHK across all jobs per activity                   |
| `aggregate_whole_run_data()`    | Concatenates all activity data into whole-run arrays              |

### `plotting.py`

| Function                            | Description                                                   |
|-------------------------------------|---------------------------------------------------------------|
| `plot_mean_median_max_bars()`       | Bar chart of mean/median/max VIB & SHK per failed job        |
| `plot_activity_boxplots()`          | Boxplots of VIB & SHK per activity (failed vs. intact)       |
| `plot_shock_histogram_bar()`        | Count and percentage bar charts for shock levels             |
| `plot_shock_histogram_combined()`   | Histogram + boxplot overlay for shock                        |
| `plot_vib_histogram_bar()`          | Count and percentage bar charts for vibration levels         |
| `plot_vib_histogram_combined()`     | Histogram + boxplot overlay for vibration                    |
| `plot_vib_level_count_bars()`       | Bar chart of VIB count above thresholds (10, 12.5, 15 g)    |
| `plot_shk_level_count_bars()`       | Bar chart of SHK count above threshold (500 g)              |
| `plot_vib_scatter_motor_turbine()`  | Scatter plot of VIB counts by range, split by drive type     |
| `plot_shock_scatter_motor_turbine()`| Scatter plot of SHK counts by range, split by drive type     |
| `plot_n2_vib_shk_boxplots()`        | Boxplots of N2 rate and shock per job                        |

---

## Requirements

```
pandas
numpy
matplotlib
ctd_partitioner
IPython
```

Install dependencies:
```bash
pip install pandas numpy matplotlib ipython
```

> `ctd_partitioner` is an internal package — ensure it is available on the Python path.

---

## Usage

```bash
python main.py
```

The pipeline runs in this order:
1. Load CSV files
2. Extract and label job IDs
3. Clean data
4. Partition by activity
5. Extract valid activities
6. Run S&V analysis
7. Generate all plots

---

## Outlier Thresholds Applied

| Parameter | Valid Range |
|-----------|-------------|
| `VIB_LAT` | 0 – 100 g   |
| `SHK_LAT` | 0 – 600 g   |

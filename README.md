## MEMS_sensor_failure_analysis_for_motor_n_turbine_drive
This script performs data analysis and visualization for vibration (VIB_LAT), shock (SHK_LAT), and nitrogen flow rate (N2_RATE) data from multiple jobs. The jobs are categorized into failed jobs and intact jobs, and further grouped into motor and turbine categories. The script processes the data, performs activity-wise analysis, and generates various plots to visualize the results.

Key Steps in the Script
### Importing Libraries:

Libraries such as pandas, matplotlib, numpy, and re are imported for data manipulation, visualization, and regular expression matching.
Loading and Categorizing Data:

CSV files for failed and intact jobs are loaded from specified directories.
Job IDs are extracted using regular expressions and categorized into motor and turbine groups.
Data Cleaning and Indexing:

The data is cleaned by converting columns to numeric values and handling invalid entries.
Time columns are converted to datetime format and set as the index.
### Partitioning Data:

The data is partitioned into activities (e.g., "On Bottom Drilling", "Pull Test") using a partitioning function.
Jobs that cannot be partitioned are excluded from further analysis.
Activity-Wise Analysis:

For each activity, vibration and shock data are extracted and filtered based on thresholds (e.g., VIB_LAT < 100, SHK_LAT < 700).
Mean, median, and max values are calculated for each activity and job.
Data Grouping:

Vibration and shock data are grouped into predefined ranges (e.g., 0-10, 10-15) for both failed and intact jobs.
Separate counts are maintained for motor and turbine jobs.
Visualization:

### Bar Plots:
Vibration and shock counts for different ranges are plotted as bar charts.
Percentage distributions of vibration and shock levels are also visualized.
Scatter Plots:
Scatter plots are created to show vibration and shock counts for motor and turbine jobs across different ranges.
Box Plots:
Box plots are generated for N2_RATE, VIB_LAT, and SHK_LAT for failed and intact jobs.
Histograms:
Histograms are plotted for vibration and shock data across all jobs.
### Whole Run Analysis:

Vibration and shock data for the entire run are combined for failed and intact jobs.
Histograms and box plots are generated to visualize the overall distribution.
Key Features of the Script
Categorization:

Jobs are categorized into failed and intact jobs, and further into motor and turbine groups.
### Activity-Wise Analysis:

The script performs detailed analysis for specific activities (e.g., "Pull Test", "Wiper Trip") to understand vibration and shock behavior during these activities.
Threshold Filtering:

Data is filtered based on thresholds to exclude invalid or extreme values (e.g., VIB_LAT < 100, SHK_LAT < 700).
Statistical Analysis:

Mean, median, and max values are calculated for vibration and shock data for each activity and job.
#### Visualization:

The script generates a variety of plots (bar, scatter, box, and histogram) to visualize the data and highlight differences between failed and intact jobs.
Outputs
Bar Plots:

Vibration and shock counts for different ranges.
Percentage distributions of vibration and shock levels.
Scatter Plots:

Vibration and shock counts for motor and turbine jobs across different ranges.
Box Plots:

Distribution of N2_RATE, VIB_LAT, and SHK_LAT for failed and intact jobs.
Histograms:

Overall distribution of vibration and shock data for all jobs.
Summary Statistics:

Mean, median, and max values for vibration and shock data for each activity and job.
Applications
This script is useful for:

Failure Analysis: Identifying patterns in vibration and shock data that differentiate failed jobs from intact jobs.
Activity Monitoring: Understanding how vibration and shock levels vary across different activities.
Data Visualization: Providing clear visual insights into the behavior of vibration, shock, and nitrogen flow rate data.
Predictive Maintenance: Highlighting potential issues in motor and turbine jobs based on vibration

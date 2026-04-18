import sys
sys.path.append('..')

from config import activities, job_categories
from data_loader import (load_csv_files, extract_and_label_job_ids,
                         clean_data, partition_all_jobs, get_jobs_activities)
from snv_analysis import (build_activity_dicts, aggregate_activity_data, aggregate_whole_run_data)
from plotting import (plot_mean_median_max_bars, plot_activity_boxplots,
                      plot_shock_histogram_bar, plot_shock_histogram_combined,
                      plot_vib_histogram_bar, plot_vib_histogram_combined,
                      plot_vib_level_count_bars, plot_shk_level_count_bars,
                      plot_vib_scatter_motor_turbine, plot_shock_scatter_motor_turbine,
                      plot_n2_vib_shk_boxplots)

if __name__ == "__main__":
    # --- Load ---
    csv_files_allJobs = load_csv_files()
    job_ids, job_id_wise_csv_data = extract_and_label_job_ids(csv_files_allJobs)

    # --- Clean ---
    job_wise_cleaned_data = clean_data(job_ids, job_id_wise_csv_data)

    # --- Partition ---
    job_wise_partitioned_data, updated_job_ids = partition_all_jobs(job_ids, job_wise_cleaned_data)

    # --- Activities ---
    jobs_activities = get_jobs_activities(updated_job_ids, job_wise_partitioned_data)

    # --- S&V Analysis ---
    activity_dict_vib, activity_dict_shk = build_activity_dicts(
        activities, updated_job_ids, job_wise_partitioned_data, job_wise_cleaned_data)

    all_job_activity_vib_data, all_job_activity_shock_data = aggregate_activity_data(
        activities, updated_job_ids, activity_dict_vib, activity_dict_shk)

    vib_all_job_data, shock_all_job_data = aggregate_whole_run_data(
        activities, updated_job_ids, activity_dict_vib, activity_dict_shk)

    # --- Plots ---
    plot_mean_median_max_bars(activity_dict_vib, activity_dict_shk, job_ids)
    plot_activity_boxplots(activities, all_job_activity_vib_data, all_job_activity_shock_data)
    plot_shock_histogram_bar(shock_all_job_data)
    plot_shock_histogram_combined(shock_all_job_data)
    plot_vib_histogram_bar(vib_all_job_data)
    plot_vib_histogram_combined(vib_all_job_data)
    plot_vib_level_count_bars(job_ids, job_wise_cleaned_data)
    plot_shk_level_count_bars(job_ids, job_wise_cleaned_data)
    plot_vib_scatter_motor_turbine(job_wise_cleaned_data)
    plot_shock_scatter_motor_turbine(job_wise_cleaned_data)
    plot_n2_vib_shk_boxplots(job_ids, job_wise_cleaned_data)

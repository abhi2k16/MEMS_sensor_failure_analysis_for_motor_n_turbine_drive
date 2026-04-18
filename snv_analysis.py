import pandas as pd
import numpy as np
from config import job_categories


def mems_SnV_analysis_alljobs(activities, partitioner_data, main_data):
    """
    activities:       list of activity names e.g. ['Wiper Trip']
    partitioner_data: partitioned DataFrame for a particular job
    main_data:        raw cleaned DataFrame for a particular job
    Returns: (vib_list_all, shock_list_all, vib_level_count)
    """
    vib_list_all = {}
    shock_list_all = {}
    vib_level_count = {}
    for activity in activities:
        vib_level_count[activity] = []
        vib_list_all[activity] = []
        shock_list_all[activity] = []

        ctd_partitioner = partitioner_data[['Start Time', 'End Time', 'Activity']]
        data_m = main_data[['TIME', 'VIB_LAT', 'SHK_LAT', 'FLWI', 'DEPT', 'BVEL']]

        ctd_partitioner_data_df = pd.concat([ctd_partitioner], ignore_index=False)
        ctd_main_dataset_df = pd.concat([data_m], ignore_index=False)
        ctd_main_dataset_df['TIME'] = pd.to_datetime(ctd_main_dataset_df['TIME'])

        activity_data = ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
        activity_data_append = []
        for _, s in activity_data.iterrows():
            data_activity = ctd_main_dataset_df[s['Start Time']:s['End Time']]
            activity_data_append.append(data_activity)

        if activity_data_append:
            activity_data_df = pd.concat(activity_data_append, ignore_index=False)
        else:
            activity_data_df = pd.DataFrame(columns=['TIME', 'VIB_LAT', 'SHK_LAT', 'FLWI', 'DEPT', 'BVEL'])

        activity_data_df[['SHK_LAT']] = activity_data_df[['SHK_LAT']].mask(
            (activity_data_df['SHK_LAT'] > 600.0) | (activity_data_df['SHK_LAT'] < 0))
        activity_data_df[['VIB_LAT']] = activity_data_df[['VIB_LAT']].mask(
            (activity_data_df['VIB_LAT'] > 100.0) | (activity_data_df['VIB_LAT'] < 0))
        activity_data_df = activity_data_df.dropna()

        for label in [10, 15, 20]:
            vib_level_count[activity].append(activity_data_df[activity_data_df['VIB_LAT'] > label]['VIB_LAT'].count())
        vib_level_count[activity].append(activity_data_df['VIB_LAT'].count())
        vib_list_all[activity].append(activity_data_df['VIB_LAT'].values)
        shock_list_all[activity].append(activity_data_df['SHK_LAT'].values)

    return vib_list_all, shock_list_all, vib_level_count


def mean_median_max_values(parameter_data, activities, job_ids):
    """
    parameter_data: activity_dict_vib or activity_dict_shk
    activities:     list of activity names
    job_ids:        dict of job_ids per job_type
    Returns: (mean_value, median_value, max_value)
    """
    mean_value = {}
    median_value = {}
    max_value = {}
    for job_type in [job_categories[0]]:
        for activity in activities:
            mean_value[activity] = {}
            median_value[activity] = {}
            max_value[activity] = {}
            for job_id in job_ids.get(job_type):
                data_v = parameter_data.get(job_type).get(activity).get(job_id)
                try:
                    mean_value[activity][job_id] = list(np.mean(data_v, axis=1))
                    median_value[activity][job_id] = list(np.median(data_v, axis=1))
                    max_value[activity][job_id] = list(np.max(data_v, axis=1))
                except:
                    mean_value[activity][job_id] = [np.nan]
                    median_value[activity][job_id] = [np.nan]
                    max_value[activity][job_id] = [np.nan]
    return mean_value, median_value, max_value


def build_activity_dicts(activities, updated_job_ids, job_wise_partitioned_data, job_wise_cleaned_data):
    activity_dict_vib = {}
    activity_dict_shk = {}
    for job_type in job_categories:
        activity_dict_vib[job_type] = {}
        activity_dict_shk[job_type] = {}
        for activity in activities:
            activity_dict_vib[job_type][activity] = {}
            activity_dict_shk[job_type][activity] = {}
            for job_id in updated_job_ids.get(job_type):
                activity_dict_vib[job_type][activity][job_id] = []
                activity_dict_shk[job_type][activity][job_id] = []
                vib_dict, shk_dict, _ = mems_SnV_analysis_alljobs(
                    activities=[activity],
                    partitioner_data=job_wise_partitioned_data.get(job_type).get(job_id),
                    main_data=job_wise_cleaned_data.get(job_type).get(job_id)
                )
                try:
                    activity_dict_shk[job_type][activity][job_id].extend(shk_dict[activity])
                    activity_dict_vib[job_type][activity][job_id].extend(vib_dict[activity])
                except:
                    activity_dict_shk[job_type][activity][job_id].append(np.nan)
                    activity_dict_vib[job_type][activity][job_id].append(np.nan)
    return activity_dict_vib, activity_dict_shk


def aggregate_activity_data(activities, updated_job_ids, activity_dict_vib, activity_dict_shk):
    all_job_activity_vib_data = {}
    all_job_activity_shock_data = {}
    for job_type in job_categories:
        all_job_activity_vib_data[job_type] = {}
        all_job_activity_shock_data[job_type] = {}
        for activity in activities:
            all_job_activity_vib_data[job_type][activity] = []
            all_job_activity_shock_data[job_type][activity] = []
            for job_id in updated_job_ids.get(job_type):
                all_job_activity_vib_data[job_type][activity].extend(
                    activity_dict_vib.get(job_type).get(activity).get(job_id))
                all_job_activity_shock_data[job_type][activity].extend(
                    activity_dict_shk.get(job_type).get(activity).get(job_id))
    return all_job_activity_vib_data, all_job_activity_shock_data


def aggregate_whole_run_data(activities, updated_job_ids, activity_dict_vib, activity_dict_shk):
    all_job_vib_data = {jt: [] for jt in job_categories}
    all_job_shock_data = {jt: [] for jt in job_categories}
    for job_type in job_categories:
        for activity in activities:
            for job_id in updated_job_ids.get(job_type):
                all_job_vib_data[job_type].extend(activity_dict_vib.get(job_type).get(activity).get(job_id))
                all_job_shock_data[job_type].extend(activity_dict_shk.get(job_type).get(activity).get(job_id))
    vib_all = [np.concatenate(all_job_vib_data['failed_job_csv']),
               np.concatenate(all_job_vib_data['intact_job_csv'])]
    shock_all = [np.concatenate(all_job_shock_data['failed_job_csv']),
                 np.concatenate(all_job_shock_data['intact_job_csv'])]
    return vib_all, shock_all

if __name__ == "__main__":
    print("This module contains functions for SnV analysis and is not meant to be run directly.")   
    pass


import re
import glob
import os
import pandas as pd
from ctd_partitioner.activity_partitioners import partition_data
from config import directory_failed_jobs, directory_intact_jobs, job_ids_motor_turbine, job_categories, columns_of_interest


def load_csv_files():
    csv_files_failed = glob.glob(os.path.join(directory_failed_jobs, "*.csv"))
    csv_files_intact = glob.glob(os.path.join(directory_intact_jobs, "*.csv"))
    return {"failed_job_csv": csv_files_failed, "intact_job_csv": csv_files_intact}


def extract_and_label_job_ids(csv_files_allJobs):
    job_ids = {}
    job_id_wise_csv_data = {}
    for job_type in job_categories:
        job_ids[job_type] = []
        job_id_wise_csv_data[job_type] = {}
        for files in csv_files_allJobs.get(job_type):
            match = re.search(r'AllJobsData\\(O\.\d+\.\d+-\d+)_concat_data\.csv', files)
            if match:
                job_id = match.group(1)
                job_id = f'{job_id}[M]' if job_id in job_ids_motor_turbine['motor'] else f'{job_id}[T]'
                print(job_id)
                job_ids[job_type].append(job_id)
                job_id_wise_csv_data[job_type][job_id] = pd.read_csv(files)
            else:
                print("Job ID not found")
    return job_ids, job_id_wise_csv_data


def clean_data(job_ids, job_id_wise_csv_data):
    job_wise_cleaned_data = {}
    for job_type in job_categories:
        job_wise_cleaned_data[job_type] = {}
        for job_id in job_ids.get(job_type):
            print(job_id)
            data = job_id_wise_csv_data.get(job_type).get(job_id)
            data['TIME'] = pd.to_datetime(data['DateTime'])
            data.index = data['TIME']
            data = data[columns_of_interest]
            for col in columns_of_interest:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            job_wise_cleaned_data[job_type][job_id] = data
    return job_wise_cleaned_data


def partition_all_jobs(job_ids, job_wise_cleaned_data):
    job_wise_partitioned_data = {}
    non_partitioned_job_id = []
    for job_type in job_categories:
        job_wise_partitioned_data[job_type] = {}
        for job_id in job_ids.get(job_type):
            try:
                data_to_partition = job_wise_cleaned_data.get(job_type).get(job_id)
                partition_df = partition_data(data_to_partition)
                pd.set_option('display.max_rows', data_to_partition.shape[0])
                partition_df.index = pd.to_datetime(partition_df['Start Time'])
                partition_df['Start Time'] = pd.to_datetime(partition_df['Start Time'])
                partition_df['End Time'] = pd.to_datetime(partition_df['End Time'])
                job_wise_partitioned_data[job_type][job_id] = partition_df.sort_index()
            except:
                non_partitioned_job_id.append(job_id)
                print(f'Job Id: {job_id} is not partitioned')
    updated_job_ids = {
        job_type: [jid for jid in job_ids.get(job_type) if jid not in non_partitioned_job_id]
        for job_type in job_categories
    }
    return job_wise_partitioned_data, updated_job_ids


def get_jobs_activities(updated_job_ids, job_wise_partitioned_data):
    jobs_activities = {}
    for job_type in job_categories:
        jobs_activities[job_type] = {}
        for job_id in updated_job_ids.get(job_type):
            activity_list = job_wise_partitioned_data.get(job_type).get(job_id)['Activity'].unique().tolist()
            jobs_activities[job_type][job_id] = activity_list
            print(job_id)
    return jobs_activities

if __name__ == "__main__":
    print("This module contains functions for loading and processing data and is not meant to be run directly.")
    pass
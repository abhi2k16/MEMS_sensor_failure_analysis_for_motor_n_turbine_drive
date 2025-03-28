#%%
import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os
from IPython.display import display, HTML
from ctd_partitioner.state_detection import compute_on_well_construction_activity_states
from ctd_partitioner.activity_partitioners import partition_data
#%% ################################# Activity-Wise Analysis For All Jobs W/WO fail Combined ####################################### 
directory_intact_jobs = "D:/2025/DataAnalysisShockMEMSSensor/DataWOFail/AllJobsData"
directory_failed_jobs = "D:/2025/DataAnalysisShockMEMSSensor/Data/AllJobsData"
csv_files_failed_Jobs = glob.glob(os.path.join(directory_failed_jobs, "*.csv"))
csv_files_intact_Jobs = glob.glob(os.path.join(directory_intact_jobs, "*.csv"))
csv_files_allJobs = {"failed_job_csv": csv_files_failed_Jobs,
                     "intact_job_csv": csv_files_intact_Jobs}
#%% ####################### Job ids grouped in to motor and turbine ###########################
job_ids_motor_turbine = {'motor': ['O.1048592.41-7','O.1048592.59-6','O.1048592.107-3','O.1048592.107-6'
                                   ,'O.1048592.107-7','O.1048592.107-9','O.1048592.107-10','O.1048592.133-1'],
                                   'turbine':['O.1048592.72-18','O.1048592.99-5','O.1048592.59-9','O.1048592.59-10',
                                              'O.1048592.110-2','O.1048592.110-4','O.1048592.110-5','O.1048592.110-6',
                                              'O.1048592.133-5']}
#%%
job_categories = ["failed_job_csv","intact_job_csv"]
job_ids = {}
job_id_wise_csv_data = {}
for job_type in job_categories:
    job_ids[job_type] = []
    job_id_wise_csv_data[job_type] = {}
    for files in csv_files_allJobs.get(job_type):
        #print(files)
        match = re.search(r'AllJobsData\\(O\.\d+\.\d+-\d+)_concat_data\.csv', files)
        if match:
            job_id = match.group(1)
            # categorising jobs in to motor and turbine
            job_id = f'{job_id}[M]' if job_id in job_ids_motor_turbine['motor'] else f'{job_id}[T]'
            print(job_id)
            job_ids[job_type].append(job_id)
            job_id_wise_csv_data[job_type][job_id] = pd.read_csv(files)
        else:
            print("Job ID not found")
#%%  ################## Cleaning and indexing the dataset for all job_ids ########################### 
job_wise_cleaned_data = {}
for job_type in job_categories:
    job_wise_cleaned_data[job_type] = {}
    for job_id in job_ids.get(job_type):
        # categorising jobs in to motor and turbine
        #job_id = f'{job_id}[M]' if job_id in job_ids_motor_turbine['motor'] else f'{job_id}[T]'
        print(job_id)
        data = job_id_wise_csv_data.get(job_type).get(job_id)
        data['TIME'] = pd.to_datetime(data['DateTime'])
        data.index = data['TIME']
        columns_of_interest = ['TIME','BVEL','CT_WGT','DEPT', 'HDTH','FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']
        data = data[['TIME','BVEL','CT_WGT','DEPT','HDTH', 'FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']]
        #print(data.head())
        for column_name in columns_of_interest:
            data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
        job_wise_cleaned_data[job_type][job_id] = data
#%% ###################### Partitioning the dataset for all job_ids ############################# 
job_wise_partitioned_data = {}
non_partitioned_job_id = []
for job_type in job_categories:
    job_wise_partitioned_data[job_type] = {}
    for job_id in job_ids.get(job_type):
        try:
            data_to_partition = job_wise_cleaned_data.get(job_type).get(job_id)
            partition_df = partition_data(data_to_partition)
            pd.set_option('display.max_rows', data.shape[0])
            partition_df.index = pd.to_datetime(partition_df['Start Time'])
            partition_df['Start Time'] = pd.to_datetime(partition_df['Start Time'])
            partition_df['End Time'] = pd.to_datetime(partition_df['End Time'])
            partitions_sort = partition_df.sort_index()
            job_wise_partitioned_data[job_type][job_id] = partitions_sort
        except:
            non_partitioned_job_id.append(job_id)
            print(f'Job Id: {job_id} is not partitioned')
updated_job_ids = {}
for job_type in job_categories:
    updated_job_ids[job_type] = [job_id for job_id in job_ids.get(job_type) if job_id not in non_partitioned_job_id]
# %% Job wise valid activity 
jobs_activities = {}
for job_type in job_categories:
    jobs_activities[job_type] = {}
    for job_id in updated_job_ids.get(job_type):
        partition_data_df_activity = job_wise_partitioned_data.get(job_type).get(job_id)['Activity'].unique().tolist()
        jobs_activities[job_type][job_id] = []
        jobs_activities[job_type][job_id].extend(partition_data_df_activity)
        print(job_id)
        #print(partition_data_df_activity)
#%% ##########function for get activity wise S&V data for given jobs ###############
def mems_SnV_analysis_alljobs(activities, partitioner_data, main_data):
    """---------------------------------------input variables----------------------------------------------
    activities:  ['Wiper Trip'] #list of activities
    partitioner_data :  Dataset after partition e.g., "ctd_partitions_data" # data for a particular job
    main data :  non-partition data e.g., main_data # dataset for a particular job
    ----------------------------------------------------------------------------------------------------"""
    vib_list_all  = {}
    shock_list_all = {}
    vib_level_count = {}
    for activity in activities:
        vib_level_count[activity] = []
        vib_list_all[activity] = []
        shock_list_all[activity] = []
        ctd_partitioner_data = []
        ctd_main_dataset = []
        ctd_partitions = partitioner_data
        ctd_partitioner = ctd_partitions[['Start Time','End Time','Activity']]
        ctd_partitioner_data.append(ctd_partitioner)
        main_dataset = main_data
        data_m = main_dataset[['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL']]
        ctd_main_dataset.append(data_m)
        ctd_partitioner_data_df = pd.concat(ctd_partitioner_data,ignore_index = False)
        ctd_main_dataset_df = pd.concat(ctd_main_dataset, ignore_index = False)
        activity_data=ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
        activity_data_df = pd.DataFrame(columns = ['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL'])
        ctd_main_dataset_df['TIME'] = pd.to_datetime(ctd_main_dataset_df['TIME'])
        activity_data_append = []
        for r,s in activity_data.iterrows():
            start_time = s['Start Time']
            end_time = s['End Time']
            data_activity = ctd_main_dataset_df[start_time:end_time]
            activity_data_append.append(data_activity)
        if activity_data_append:
            activity_data_df = pd.concat(activity_data_append,ignore_index = False)
        else:
            pass
        activity_data_df[['SHK_LAT']] = activity_data_df[
            ['SHK_LAT']].mask((activity_data_df['SHK_LAT'] > 600.0)|
                              (activity_data_df['SHK_LAT'] < 0))
        activity_data_df[['VIB_LAT']] = activity_data_df[
            ['VIB_LAT']].mask((activity_data_df['VIB_LAT'] > 100.0)|
                              (activity_data_df['VIB_LAT'] < 0))
        activity_data_df = activity_data_df.dropna()
        vib_label = [10, 15, 20]
        for label in vib_label:
            count = activity_data_df[activity_data_df['VIB_LAT']> label]['VIB_LAT'].count()
            vib_level_count[activity].append(count)
        
        vib_level_count[activity].append(activity_data_df['VIB_LAT'].count())
        vib_list_all[activity].append(activity_data_df['VIB_LAT'].values)   
        shock_list_all[activity].append(activity_data_df['SHK_LAT'].values)
    return (vib_list_all, shock_list_all, vib_level_count)
#%% ###########Activity wise S&V box plot for all jobs ###################
activities = ['On Bottom Drilling','Pull Test', 
              'Wiper Trip','Trip In Run', 'Trip Out Run'] 
# only these five activities includes almost the whole run because, drill run is combination of these 3 activity
# the other activity not that much relevant, such as N2 rate increasing/decreasing  
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
            if activity in activities:
                vib_dict, shk_dict,_ = mems_SnV_analysis_alljobs(activities=[activity], 
                                                    partitioner_data=job_wise_partitioned_data.get(job_type).get(job_id),
                                                        main_data=job_wise_cleaned_data.get(job_type).get(job_id))
                try:
                    activity_dict_shk[job_type][activity][job_id].extend(shk_dict[activity])
                    activity_dict_vib[job_type][activity][job_id].extend(vib_dict[activity])
                except:
                    activity_dict_shk[job_type][activity][job_id].append(np.nan)
                    activity_dict_vib[job_type][activity][job_id].append(np.nan)
            else:
                activity_dict_vib[job_type][activity][job_id].append(np.nan)
                activity_dict_shk[job_type][activity][job_id].append(np.nan)
                print(f"For job id: {job_id}, {activity} is not partitioned list filled with NaN")
#%% ######### Ploting Mean, max and median value for S&V for all failed jobs ##############
def mean_median_max_values(parameter_data, activities):
    """ I/P:parameter_data = activity_dict_data for vib/shock,
        activities: list of activity
        O/P: mean,median, max value in dictionary for mean{activity{job_id}:[]}"""
    mean_value = {}
    median_value = {}
    max_value = {}
    for job_type in [job_categories[0]]:
        for activity in activities:
            mean_value[activity] = {}
            median_value[activity] = {}
            max_value[activity] = {}
            for job_id in job_ids.get(job_type):
                mean_value[activity][job_id] = []
                median_value[activity][job_id] = []
                max_value[activity][job_id] = []
                data_v = parameter_data.get(job_type).get(activity).get(job_id)
                try:
                    mean = np.mean(data_v,axis=1)
                    median = np.median(data_v, axis=1)
                    max = np.max(data_v, axis= 1)
                    mean_value[activity][job_id].extend(mean)
                    median_value[activity][job_id].extend(median)
                    max_value[activity][job_id].extend(max)
                except:
                    mean = median= max = np.nan 
                    mean_value[activity][job_id].append(mean)
                    median_value[activity][job_id].append(median)
                    max_value[activity][job_id].append(max)
    return (mean_value, mean_value, max_value)
#%% ########## Plot mean, median, and max vib/shock values as bar charts for each failed job ############
plt.rcParams.update({'font.size':12})
parameters = ['VIB_LAT','SHK_LAT']
for parameter in parameters:
    parameter_data = activity_dict_vib if parameter == 'VIB_LAT' else activity_dict_shk
    mean_value,median_value,max_value = mean_median_max_values(parameter_data=parameter_data,
                                                                  activities=activities)
    label = 'Lat. Vib' if parameter == 'VIB_LAT' else 'Lat. Shock'
    for activity in mean_value.keys():
        failed_job_ids = list(mean_value[activity].keys())
        mean_values = list(mean_value[activity].values())
        mean_values = np.array([item for sublist in mean_values for item in sublist])
        median_values = list(median_value[activity].values())
        median_values = np.array([item for sublist in median_values for item in sublist])
        max_values = list(max_value[activity].values())
        max_values = np.array([item for sublist in max_values for item in sublist])
        x = np.arange(len(failed_job_ids))  # the label locations
        width = 0.2  # the width of the bars
        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width, mean_values, width, label='Mean')
        rects2 = ax.bar(x, median_values, width, label='Median')
        rects3 = ax.bar(x + width, max_values, width, label='Max')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Job ID')
        ax.set_ylabel(f'{label}[g]')
        ax.set_title(f'{activity} - Mean, Median, and Max {label} for failed Job')
        ax.set_xticks(x)
        ax.set_xticklabels(failed_job_ids, rotation=75)
        ax.legend()
        # Add value labels on top of the bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        add_labels(rects1)
        #add_labels(rects2)
        add_labels(rects3)
        fig.tight_layout()
        plt.show()
#%% ##########activity wise all data for all failed and intact jobs ######################
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
#%%
for activity in activities:
    activity_wise_vib_all_data = [np.concatenate(all_job_activity_vib_data.get('failed_job_csv').get(activity)),
                                np.concatenate(all_job_activity_vib_data.get('intact_job_csv').get(activity))]
    activity_wise_shock_all_data = [np.concatenate(all_job_activity_shock_data.get('failed_job_csv').get(activity)),
                                np.concatenate(all_job_activity_shock_data.get('intact_job_csv').get(activity))]
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(5,3))
    plt.boxplot(activity_wise_vib_all_data, patch_artist=True, labels = ['Failed Jobs','Success Jobs'],
            boxprops=dict(facecolor='blue'),
                flierprops=dict(marker='o', color='red'))
    plt.ylabel('Value [g]')
    plt.grid()
    plt.suptitle(f"Activity: [{activity}] Lat. Vib." ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job Type',fontsize = 15)
    plt.show()
    plt.figure(figsize=(5,3))
    plt.boxplot(activity_wise_shock_all_data, patch_artist=True, labels = ['Failed Jobs','Success Jobs'],
            boxprops=dict(facecolor='blue'),
                flierprops=dict(marker='o', color='red'))
    plt.ylabel('Value [g]')
    plt.grid()
    plt.suptitle(f"Activity: [{activity}] Lat. Shock" ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job Type',fontsize = 15)
#%% ######## combining whole run data for all failed and intact jobs ##################
all_job_vib_data = {}
all_job_shock_data = {}
for job_type in job_categories:
    all_job_vib_data[job_type] = []
    all_job_shock_data[job_type] = []
    for activity in activities:
        for job_id in updated_job_ids.get(job_type):
            all_job_vib_data[job_type].extend(
                activity_dict_vib.get(job_type).get(activity).get(job_id))
            all_job_shock_data[job_type].extend(
                activity_dict_shk.get(job_type).get(activity).get(job_id))
#%%############### combining whole run data for all failed and intact jobs #################
vib_all_job_data = [np.concatenate(all_job_vib_data.get('failed_job_csv')),
                            np.concatenate(all_job_vib_data.get('intact_job_csv'))]
shock_all_job_data = [np.concatenate(all_job_shock_data.get('failed_job_csv')),
                            np.concatenate(all_job_shock_data.get('intact_job_csv'))]
#%%#####################################################################################
################################ PLOTS FOR SHOCK #######################################
########################################################################################

######## Bar plot for the different lat. Shock level for all jobs ######################
bins = np.arange(0, 600, 50)
failed_job_hist_shk, _ = np.histogram(shock_all_job_data[0], bins=bins)
intact_job_hist_shk, _ = np.histogram(shock_all_job_data[1], bins=bins)
intact_job_hist_shk = np.array([np.nan if x == 0 else x for x in intact_job_hist_shk])
bin_labels = [f'{i}-{i+50}' for i in bins[:-1]]
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(bin_labels))
bars1 = plt.bar(index, failed_job_hist_shk, bar_width, label='Failed Jobs', color='red', edgecolor='black')
bars2 = plt.bar(index + bar_width, intact_job_hist_shk, bar_width, label='Success Jobs', color='green', edgecolor='black')
plt.xlabel('Lat. Shock [g]', fontsize=14)
plt.ylabel('Count [log scale]', fontsize=14)
plt.title('Lat. Shock for all all jobs', fontsize=16)
plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
plt.yscale('log')
plt.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.ylim([0, max(max(failed_job_hist), max(intact_job_hist)) + 2])
plt.tight_layout()
plt.show()
# %% ####Bar plot for the percentage of time wise lat. shock level for all jobs ################
failed_job_hist_pecent_shk = failed_job_hist_shk/sum(failed_job_hist_shk)*100
intact_job_hist_shk_2 = [0 if np.isnan(x) else x for x in intact_job_hist_shk]
intact_job_hist_pecent_shk = intact_job_hist_shk_2/sum(intact_job_hist_shk_2)*100
plt.figure(figsize=(12, 6))
bar_width = 0.40
index = np.arange(len(bin_labels))
bars1 = plt.bar(index, failed_job_hist_pecent_shk, bar_width, label='Failed Jobs', color='red', edgecolor='black')
bars2 = plt.bar(index + bar_width, intact_job_hist_pecent_shk, bar_width, label='Success Jobs', color='green', edgecolor='black')
plt.xlabel('Lat. Shock [g]', fontsize=14)
plt.ylabel('Percentage of Time', fontsize=14)
plt.title('Lat. Shock for all all jobs', fontsize=16)
plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
plt.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = round(bar.get_height(), 2)
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%###### Histohram plot for the Shock value for different range for all jobs ##########
xtick = np.arange(0,600,50)
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 7), sharex=True)
# Histogram (Top Plot)
ax[0].hist(shock_all_job_data[0], bins=100, color='red', alpha=1, label="Failed Jobs", density=False)
ax[0].hist(shock_all_job_data[1], bins=100, color='green', alpha=0.6, label="Success Jobs", density=False)
ax[0].legend()
ax[0].set_ylabel("Count [log scale]")
ax[0].set_title("Lat. Shock for All Jobs")
ax[0].set_yscale('log',base = 10)
ax[0].grid(True)
# Box Plot (Bottom Plot)
ax[1].boxplot([shock_all_job_data[0], shock_all_job_data[1]], vert=False, patch_artist=True,
              boxprops=dict(facecolor="blue", alpha=1),
              medianprops=dict(color="black"),
              labels=["Failed Jobs", "Success Jobs"])
ax[1].set_xlabel("Lat. Shock [g]")
plt.tight_layout()
plt.xlim([0, 65])
plt.xticks(xtick)
plt.grid(axis='x')
plt.show()
#%%#####################################################################################
################################ PLOTS FOR VIBRATION ###################################
########################################################################################
######### Bar plot for the different lat. vibration level for all jobs #################
bins = np.arange(0,70,10)
failed_job_hist_vib,_ = np.histogram(vib_all_job_data[0], bins=bins)
success_job_hist_vib,_ = np.histogram(vib_all_job_data[1],bins=bins)
success_job_hist_vib = [np.nan if x == 0 else x for x in success_job_hist_vib]
bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(bin_labels))
bars1 = plt.bar(index, failed_job_hist_vib, bar_width, label='Failed Jobs', color='red', edgecolor='black')
bars2 = plt.bar(index + bar_width, success_job_hist_vib, bar_width, label='Success Jobs', color='green', edgecolor='black')
plt.xlabel('Lat. Vib. [g]', fontsize=14)
plt.ylabel('Count [log scale]', fontsize=14)
plt.title('Lat. Vib. for all all jobs', fontsize=16)
plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
plt.yscale('log')
plt.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %% ####Bar plot for the percentage of time wise lat. vibration level for all jobs ################
failed_job_hist_pecent_vib = failed_job_hist_vib/sum(failed_job_hist_vib)*100
success_job_hist_vib_2 = [0 if np.isnan(x) else x for x in success_job_hist_vib]
success_job_hist_pecent_vib = success_job_hist_vib_2/sum(success_job_hist_vib_2)*100
plt.figure(figsize=(12, 6))
bar_width = 0.40
index = np.arange(len(bin_labels))
bars1 = plt.bar(index, failed_job_hist_pecent_vib, bar_width, label='Failed Jobs', color='red', edgecolor='black')
bars2 = plt.bar(index + bar_width, success_job_hist_pecent_vib, bar_width, label='Success Jobs', color='green', edgecolor='black')
plt.xlabel('Lat. Vib. [g]', fontsize=14)
plt.ylabel('Percentage of Time', fontsize=14)
plt.title('Lat. Vib. for all all jobs', fontsize=16)
plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
plt.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = round(bar.get_height(), 2)
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%###### Histohram plot for the Vibration value for different range for all jobs ##########
xtick = np.arange(0,65,5)
fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 7), sharex=True)
# Histogram (Top Plot)
ax[0].hist(vib_all_job_data[0], bins=100, color='red', alpha=1, label="Failed Jobs", density=False)
ax[0].hist(vib_all_job_data[1], bins=100, color='green', alpha=0.6, label="Success Jobs", density=False)
ax[0].legend()
ax[0].set_ylabel("Count [log scale]")
ax[0].set_title("Lat. Vib. for All Jobs")
ax[0].set_yscale('log',base = 10)
ax[0].grid(True)
ax[1].boxplot([vib_all_job_data[0], vib_all_job_data[1]], vert=False, patch_artist=True,
              boxprops=dict(facecolor="blue", alpha=1),
              medianprops=dict(color="black"),
              labels=["Failed Jobs", "Success Jobs"])
ax[1].set_xlabel("Lat. Vib. [g]")
plt.tight_layout()
plt.xlim([0, 65])
plt.xticks(xtick)
plt.grid(axis='x')
plt.show()
#%% ################################################################################################################
##############################  Analysis for Whole Run For All Jobs W/WO fail Combined #############################
####################################################################################################################
#if 'O.1048592.59-6' in job_ids['failed_job_csv']:
#    job_ids['failed_job_csv'].remove('O.1048592.59-6','O.1048592.107-7') # Removing job_id: 'O.1048592.59-6' bcs have incomplete dataset
#job_ids_to_remove = ['O.1048592.59-6', 'O.1048592.107-7'] # job_ids: 'O.1048592.107-7' with exeptionally high S&V values
# Remove job IDs from 'failed_job_csv' if they exist
#job_ids['failed_job_csv'] = [job_id for job_id in job_ids['failed_job_csv'] if job_id not in job_ids_to_remove]
############################# Bar plot for Vibration level count ############################
vib_level = [10,12.5,15]
for level in vib_level: 
    job_type_wise_vib_count = {}
    for job_type in job_categories:
        job_type_wise_vib_count[job_type] = {}
        for job_id in job_ids.get(job_type):
            job_type_wise_vib_count[job_type][job_id] = []
            job_data = job_wise_cleaned_data.get(job_type).get(job_id)
            vib_count = job_data[job_data['VIB_LAT']>level]['VIB_LAT'].count()
            job_type_wise_vib_count[job_type][job_id].extend([vib_count])
    print(job_type_wise_vib_count)
    count_data = job_type_wise_vib_count
    failed_jobs = list(count_data['failed_job_csv'].keys())
    failed_values = [v[0] for v in count_data['failed_job_csv'].values()]
    intact_jobs = list(count_data['intact_job_csv'].keys())
    intact_values = [v[0] for v in count_data['intact_job_csv'].values()]
    failed_x = range(len(failed_jobs))
    intact_x = range(len(failed_jobs), len(failed_jobs) + len(intact_jobs))
    plt.figure(figsize=(12, 6))
    # Plot bars
    bars_failed = plt.bar(failed_x, failed_values, color='red', label='Failed Jobs')
    bars_intact = plt.bar(intact_x, intact_values, color='green', label='Success Jobs')

    for bar in bars_failed:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars_intact:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.rcParams.update({'font.size': 15})
    all_labels = failed_jobs + intact_jobs
    plt.xticks(ticks=list(failed_x) + list(intact_x), labels=all_labels, rotation=75)
    plt.xlabel('Job ID')
    plt.ylabel('Count [Log Scale]')
    plt.yscale('log',base = 2)
    plt.title(f'Failed vs. Success Job Vib. [>{level }(g)] Counts')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
#%% ############################# Bar plot for Shock level count ############################ 
shk_level = [500]
for level in shk_level: 
    job_type_wise_shk_count = {}
    for job_type in job_categories:
        job_type_wise_shk_count[job_type] = {}
        for job_id in job_ids.get(job_type):
            job_type_wise_shk_count[job_type][job_id] = []
            job_data = job_wise_cleaned_data.get(job_type).get(job_id)
            shk_count = job_data[job_data['SHK_LAT']>level]['SHK_LAT'].count()
            job_type_wise_shk_count[job_type][job_id].extend([shk_count])
    print(job_type_wise_shk_count)
    count_data = job_type_wise_shk_count
    failed_jobs = list(count_data['failed_job_csv'].keys())
    failed_values = [v[0] for v in count_data['failed_job_csv'].values()]
    intact_jobs = list(count_data['intact_job_csv'].keys())
    intact_values = [v[0] for v in count_data['intact_job_csv'].values()]
    failed_x = range(len(failed_jobs))
    intact_x = range(len(failed_jobs), len(failed_jobs) + len(intact_jobs))
    plt.figure(figsize=(12, 6))
    # Plot bars
    bars_failed = plt.bar(failed_x, failed_values, color='red', label='Failed Jobs')
    bars_intact = plt.bar(intact_x, intact_values, color='green', label='Success Jobs')

    for bar in bars_failed:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    for bar in bars_intact:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height()}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.rcParams.update({'font.size': 15})
    all_labels = failed_jobs + intact_jobs
    plt.xticks(ticks=list(failed_x) + list(intact_x), labels=all_labels, rotation=75)
    plt.xlabel('Job ID')
    plt.ylabel('Count [Log Scale]')
    plt.yscale('log', base = 2)
    plt.title(f'Failed vs. Success Job shock [>{level }(g)] Counts')
    plt.legend()
    plt.grid()
    #plt.tight_layout()
    plt.show()
#%%
failed_job_vib_data = []
success_job_vib_data = []
for job_type in job_categories:
    for job_id in job_ids.get(job_type):
        if job_type == job_categories[0]:
            failed_job_vib_data.extend(job_wise_cleaned_data[job_type][job_id]['VIB_LAT'].values)
        else:
            success_job_vib_data.extend(job_wise_cleaned_data[job_type][job_id]['VIB_LAT'].values)
#%% ############# Grouping the data based on the vibration level ################
bins = [0, 10,15, 20, np.inf]
vib_groups = ['0-10','10-15','15-20','>20']
failed_job_vib_counts = {group: {'motor': [], 'turbine': []} for group in vib_groups}
success_job_vib_counts = {group: {'motor': [], 'turbine': []} for group in vib_groups}
for job_id in job_wise_cleaned_data['failed_job_csv']:
    job_data = job_wise_cleaned_data['failed_job_csv'][job_id]['VIB_LAT']
    counts, _ = np.histogram(job_data, bins=bins)
    for i,group in enumerate(vib_groups):
        if job_id[-2] == 'M':
            failed_job_vib_counts[group]['motor'].append(counts[i])
        else:
            failed_job_vib_counts[group]['turbine'].append(counts[i])
for job_id in job_wise_cleaned_data['intact_job_csv']:
    job_data = job_wise_cleaned_data['intact_job_csv'][job_id]['VIB_LAT']
    counts, _ = np.histogram(job_data, bins=bins)
    for i,group in enumerate(vib_groups):
        if job_id[-2] == 'M':
            success_job_vib_counts[group]['motor'].append(counts[i])
        else:
            success_job_vib_counts[group]['turbine'].append(counts[i])
# %% ###### scatter plot for the vibration count for each job type by vibration range #########
plt.figure(figsize=(12, 6))
for i, group in enumerate(vib_groups):
    counts = failed_job_vib_counts[group]['motor']
    counts = [count if count > 0 else 1 for count in counts]    # Replace 0 with 1 to avoid log(0)
    plt.scatter([group]*len(counts), counts, color='red',
                s=np.array(np.log10(counts))*100, label='Failed Jobs [motor]' if i == 0 else "")
for i, group in enumerate(vib_groups):
    counts = failed_job_vib_counts[group]['turbine']
    counts = [count if count > 0 else 1 for count in counts]    
    plt.scatter([group]*len(counts), counts, color='orange',
                s=np.array(np.log10(counts))*100, label='Failed Jobs [turbine]' if i == 0 else "")
for i, group in enumerate(vib_groups):
    counts = success_job_vib_counts[group]['motor']
    counts = [count if count > 0 else 1 for count in counts]
    plt.scatter([group]*len(counts), counts, color='green',
                s=np.array(np.log10(counts))*100, label='Success Jobs [motor]' if i == 0 else "")
for i, group in enumerate(vib_groups):
    counts = success_job_vib_counts[group]['turbine']
    counts = [count if count > 0 else 1 for count in counts]
    plt.scatter([group]*len(counts), counts, color='blue',
                s=np.array(np.log10(counts))*100, label='Success Jobs [turbine]' if i == 0 else "")
plt.xlabel('Lat. Vib. [g]', fontsize=14)
plt.ylabel(' Count [log scale]', fontsize=14)
plt.title('Vibration Count for Each Job Type by Vibration Range', fontsize=16)
plt.xticks(rotation=75)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.yscale('log')
plt.show()
# %% ################### Grouping the data based on the shock level ################
bins = [0,50, 100,200, 300,400,500, np.inf]
shock_groups = ['0-50','50-100','100-200','200-300','300-400','400-500','>500']
failed_job_shock_counts = {group: {'motor':[], 'turbine':[]} for group in shock_groups}
success_job_shock_counts = {group: {'motor':[], 'turbine':[]} for group in shock_groups}

for job_id in job_wise_cleaned_data['failed_job_csv']:
    """
    if job_id == 'O.1048592.107-7':
        pass
    else:"""
    job_data = job_wise_cleaned_data['failed_job_csv'][job_id]['SHK_LAT']
    counts, _ = np.histogram(job_data, bins=bins)
    for i,group in enumerate(shock_groups):
        if job_id[-2] == 'M':
            failed_job_shock_counts[group]['motor'].append(counts[i])
        else:
            failed_job_shock_counts[group]['turbine'].append(counts[i])
for job_id in job_wise_cleaned_data['intact_job_csv']:
    job_data = job_wise_cleaned_data['intact_job_csv'][job_id]['SHK_LAT']
    counts, _ = np.histogram(job_data, bins=bins)
    for i,group in enumerate(shock_groups):
        if job_id[-2] == 'M':
            success_job_shock_counts[group]['motor'].append(counts[i])
        else:
            success_job_shock_counts[group]['turbine'].append(counts[i])
#%% ###### scatter plot for the shock count for each job type by shock range #########
plt.figure(figsize=(12, 6))
for i, group in enumerate(shock_groups):
    counts = failed_job_shock_counts[group]['motor']
    count = [count if count > 0 else 1 for count in counts]    # Replace 0 with 1 to avoid log(0)
    plt.scatter([group]*len(counts), counts, color='red',
                s=np.array(np.log10(count))*100, label='Failed Jobs [motor]' if i == 0 else "")
for i, group in enumerate(shock_groups):
    counts = failed_job_shock_counts[group]['turbine']
    count = [count if count > 0 else 1 for count in counts]    
    plt.scatter([group]*len(counts), counts, color='orange',
                s=np.array(np.log10(count))*100, label='Failed Jobs [turbine]' if i == 0 else "")
for i, group in enumerate(shock_groups):
    counts = success_job_shock_counts[group]['motor']
    count = [count if count > 0 else 1 for count in counts]
    print(counts,count)
    plt.scatter([group]*len(counts), counts, color='green',
                s=np.array(np.log10(count))*100, label='Success Jobs [motor]' if i == 0 else "")
for i, group in enumerate(shock_groups):
    counts = success_job_shock_counts[group]['turbine']
    count = [count if count > 0 else 1 for count in counts]
    print(counts,count)
    plt.scatter([group]*len(counts), counts, color='blue',
                s=np.array(np.log10(count))*100, label='Success Jobs [turbine]' if i == 0 else "")
plt.xlabel('Lat. Shock [g]', fontsize=14)
plt.ylabel(' Count [log scale]', fontsize=14)
plt.title('Shock Count for Each Job Type by Shock Range', fontsize=16)
plt.xticks(rotation=75)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.yscale('log')
plt.show()
#%% ########################### plot N2_rate for each jobs #############################
########################################################################################
N2_data_jobWise = {}
vib_data_job_wise = {}
shk_data_job_wise = {}
for job_type in job_categories:
    N2_data_jobWise[job_type] = []
    vib_data_job_wise[job_type] = []
    shk_data_job_wise[job_type] = []
    for job_id in job_ids.get(job_type):
        n2_data = job_wise_cleaned_data[job_type][job_id]['N2_RATE'].to_list()
        vib_data = job_wise_cleaned_data[job_type][job_id]['VIB_LAT'].to_list()
        shk_data = job_wise_cleaned_data[job_type][job_id]['SHK_LAT'].to_list()
        n2_data = [x for x in n2_data if x >= 0]
        vib_data = [x for x in vib_data if x < 100 and x >0]
        shk_data = [x for x in shk_data if x < 700 and x >0]
        N2_data_jobWise[job_type].append(n2_data)
        vib_data_job_wise[job_type].append(vib_data)
        shk_data_job_wise[job_type].append(shk_data)
#%% ################### BoxPlot N2 for all failed and success jobs #########################
for job_type in job_categories:
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,5))

    plt.boxplot(N2_data_jobWise[job_type], patch_artist=True, labels = job_ids[job_type],
            boxprops=dict(facecolor= 'red' if job_type == 'failed_job_csv' else 'blue'),
                flierprops=dict(marker='o', color='blue' if job_type == 'failed_job_csv' else 'red'),
                medianprops=dict(color='blue' if job_type == 'failed_job_csv' else 'red', linewidth=2))
    plt.ylabel('N2 Flow Rate')
    plt.grid()
    plt.suptitle(f"Job Types: [{job_type}]" ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job id',fontsize = 15)
    plt.ylim([0,1250])
    plt.show()   

    plt.figure(figsize=(10,5))
    plt.boxplot(shk_data_job_wise[job_type], patch_artist=True, labels = job_ids[job_type],
            boxprops=dict(facecolor= 'red' if job_type == 'failed_job_csv' else 'blue'),
                flierprops=dict(marker='o', color='blue' if job_type == 'failed_job_csv' else 'red'),
                medianprops=dict(color='blue' if job_type == 'failed_job_csv' else 'red', linewidth=2))
    plt.ylabel('Lat. Vib. [g]')
    plt.grid()
    plt.suptitle(f"Job Types: [{job_type}]" ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job id',fontsize = 15)
    #plt.ylim([0,1250])
    plt.show() 
# %%

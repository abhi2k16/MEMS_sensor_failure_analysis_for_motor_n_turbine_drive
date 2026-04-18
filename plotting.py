import numpy as np
import matplotlib.pyplot as plt
from config import job_categories, activities
from snv_analysis import mean_median_max_values


def plot_mean_median_max_bars(activity_dict_vib, activity_dict_shk, job_ids):
    plt.rcParams.update({'font.size': 12})
    for parameter in ['VIB_LAT', 'SHK_LAT']:
        parameter_data = activity_dict_vib if parameter == 'VIB_LAT' else activity_dict_shk
        label = 'Lat. Vib' if parameter == 'VIB_LAT' else 'Lat. Shock'
        mean_value, median_value, max_value = mean_median_max_values(parameter_data, activities, job_ids)
        for activity in mean_value.keys():
            failed_job_ids = list(mean_value[activity].keys())
            mean_values = np.array([item for sublist in mean_value[activity].values() for item in sublist])
            median_values = np.array([item for sublist in median_value[activity].values() for item in sublist])
            max_values = np.array([item for sublist in max_value[activity].values() for item in sublist])
            x = np.arange(len(failed_job_ids))
            width = 0.2
            fig, ax = plt.subplots(figsize=(8, 5))
            rects1 = ax.bar(x - width, mean_values, width, label='Mean')
            rects2 = ax.bar(x, median_values, width, label='Median')
            rects3 = ax.bar(x + width, max_values, width, label='Max')
            ax.set_xlabel('Job ID')
            ax.set_ylabel(f'{label}[g]')
            ax.set_title(f'{activity} - Mean, Median, and Max {label} for failed Job')
            ax.set_xticks(x)
            ax.set_xticklabels(failed_job_ids, rotation=75)
            ax.legend()

            def add_labels(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate('{}'.format(round(height, 2)),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            add_labels(rects1)
            add_labels(rects3)
            fig.tight_layout()
            plt.show()


def plot_activity_boxplots(activities, all_job_activity_vib_data, all_job_activity_shock_data):
    plt.rcParams.update({'font.size': 15})
    for activity in activities:
        for data_dict, ylabel, title_suffix in [
            (all_job_activity_vib_data, 'Value [g]', 'Lat. Vib.'),
            (all_job_activity_shock_data, 'Value [g]', 'Lat. Shock')
        ]:
            plot_data = [
                np.concatenate(data_dict['failed_job_csv'][activity]),
                np.concatenate(data_dict['intact_job_csv'][activity])
            ]
            plt.figure(figsize=(5, 3))
            plt.boxplot(plot_data, patch_artist=True, labels=['Failed Jobs', 'Success Jobs'],
                        boxprops=dict(facecolor='blue'),
                        flierprops=dict(marker='o', color='red'))
            plt.ylabel(ylabel)
            plt.grid()
            plt.suptitle(f"Activity: [{activity}] {title_suffix}", fontsize=15)
            plt.tight_layout()
            plt.xticks(rotation=75)
            plt.xlabel('Job Type', fontsize=15)
            plt.show()


def plot_shock_histogram_bar(shock_all_job_data):
    bins = np.arange(0, 600, 50)
    failed_hist, _ = np.histogram(shock_all_job_data[0], bins=bins)
    intact_hist, _ = np.histogram(shock_all_job_data[1], bins=bins)
    intact_hist = np.array([np.nan if x == 0 else x for x in intact_hist])
    bin_labels = [f'{i}-{i+50}' for i in bins[:-1]]
    index = np.arange(len(bin_labels))
    bar_width = 0.35
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(index, failed_hist, bar_width, label='Failed Jobs', color='red', edgecolor='black')
    bars2 = plt.bar(index + bar_width, intact_hist, bar_width, label='Success Jobs', color='green', edgecolor='black')
    plt.xlabel('Lat. Shock [g]', fontsize=14)
    plt.ylabel('Count [log scale]', fontsize=14)
    plt.title('Lat. Shock for all jobs', fontsize=16)
    plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
    plt.yscale('log')
    plt.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    failed_pct = failed_hist / sum(failed_hist) * 100
    intact_hist_2 = [0 if np.isnan(x) else x for x in intact_hist]
    intact_pct = intact_hist_2 / sum(intact_hist_2) * 100
    plt.figure(figsize=(12, 6))
    bar_width = 0.40
    bars1 = plt.bar(index, failed_pct, bar_width, label='Failed Jobs', color='red', edgecolor='black')
    bars2 = plt.bar(index + bar_width, intact_pct, bar_width, label='Success Jobs', color='green', edgecolor='black')
    plt.xlabel('Lat. Shock [g]', fontsize=14)
    plt.ylabel('Percentage of Time', fontsize=14)
    plt.title('Lat. Shock for all jobs', fontsize=16)
    plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
    plt.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = round(bar.get_height(), 2)
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_shock_histogram_combined(shock_all_job_data):
    xtick = np.arange(0, 600, 50)
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 7), sharex=True)
    ax[0].hist(shock_all_job_data[0], bins=100, color='red', alpha=1, label="Failed Jobs")
    ax[0].hist(shock_all_job_data[1], bins=100, color='green', alpha=0.6, label="Success Jobs")
    ax[0].legend()
    ax[0].set_ylabel("Count [log scale]")
    ax[0].set_title("Lat. Shock for All Jobs")
    ax[0].set_yscale('log', base=10)
    ax[0].grid(True)
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


def plot_vib_histogram_bar(vib_all_job_data):
    bins = np.arange(0, 70, 10)
    failed_hist, _ = np.histogram(vib_all_job_data[0], bins=bins)
    success_hist, _ = np.histogram(vib_all_job_data[1], bins=bins)
    success_hist = [np.nan if x == 0 else x for x in success_hist]
    bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]
    index = np.arange(len(bin_labels))
    bar_width = 0.35
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(index, failed_hist, bar_width, label='Failed Jobs', color='red', edgecolor='black')
    bars2 = plt.bar(index + bar_width, success_hist, bar_width, label='Success Jobs', color='green', edgecolor='black')
    plt.xlabel('Lat. Vib. [g]', fontsize=14)
    plt.ylabel('Count [log scale]', fontsize=14)
    plt.title('Lat. Vib. for all jobs', fontsize=16)
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

    failed_pct = failed_hist / sum(failed_hist) * 100
    success_hist_2 = [0 if np.isnan(x) else x for x in success_hist]
    success_pct = success_hist_2 / sum(success_hist_2) * 100
    plt.figure(figsize=(12, 6))
    bar_width = 0.40
    bars1 = plt.bar(index, failed_pct, bar_width, label='Failed Jobs', color='red', edgecolor='black')
    bars2 = plt.bar(index + bar_width, success_pct, bar_width, label='Success Jobs', color='green', edgecolor='black')
    plt.xlabel('Lat. Vib. [g]', fontsize=14)
    plt.ylabel('Percentage of Time', fontsize=14)
    plt.title('Lat. Vib. for all jobs', fontsize=16)
    plt.xticks(index + bar_width / 2, bin_labels, rotation=75)
    plt.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = round(bar.get_height(), 2)
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_vib_histogram_combined(vib_all_job_data):
    xtick = np.arange(0, 65, 5)
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 7), sharex=True)
    ax[0].hist(vib_all_job_data[0], bins=100, color='red', alpha=1, label="Failed Jobs")
    ax[0].hist(vib_all_job_data[1], bins=100, color='green', alpha=0.6, label="Success Jobs")
    ax[0].legend()
    ax[0].set_ylabel("Count [log scale]")
    ax[0].set_title("Lat. Vib. for All Jobs")
    ax[0].set_yscale('log', base=10)
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


def plot_vib_level_count_bars(job_ids, job_wise_cleaned_data):
    for level in [10, 12.5, 15]:
        job_type_wise_vib_count = {}
        for job_type in job_categories:
            job_type_wise_vib_count[job_type] = {}
            for job_id in job_ids.get(job_type):
                job_data = job_wise_cleaned_data.get(job_type).get(job_id)
                vib_count = job_data[job_data['VIB_LAT'] > level]['VIB_LAT'].count()
                job_type_wise_vib_count[job_type][job_id] = [vib_count]
        failed_jobs = list(job_type_wise_vib_count['failed_job_csv'].keys())
        failed_values = [v[0] for v in job_type_wise_vib_count['failed_job_csv'].values()]
        intact_jobs = list(job_type_wise_vib_count['intact_job_csv'].keys())
        intact_values = [v[0] for v in job_type_wise_vib_count['intact_job_csv'].values()]
        failed_x = range(len(failed_jobs))
        intact_x = range(len(failed_jobs), len(failed_jobs) + len(intact_jobs))
        plt.figure(figsize=(12, 6))
        bars_failed = plt.bar(failed_x, failed_values, color='red', label='Failed Jobs')
        bars_intact = plt.bar(intact_x, intact_values, color='green', label='Success Jobs')
        for bar in list(bars_failed) + list(bars_intact):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.rcParams.update({'font.size': 15})
        plt.xticks(ticks=list(failed_x) + list(intact_x), labels=failed_jobs + intact_jobs, rotation=75)
        plt.xlabel('Job ID')
        plt.ylabel('Count [Log Scale]')
        plt.yscale('log', base=2)
        plt.title(f'Failed vs. Success Job Vib. [>{level}(g)] Counts')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


def plot_shk_level_count_bars(job_ids, job_wise_cleaned_data):
    for level in [500]:
        job_type_wise_shk_count = {}
        for job_type in job_categories:
            job_type_wise_shk_count[job_type] = {}
            for job_id in job_ids.get(job_type):
                job_data = job_wise_cleaned_data.get(job_type).get(job_id)
                shk_count = job_data[job_data['SHK_LAT'] > level]['SHK_LAT'].count()
                job_type_wise_shk_count[job_type][job_id] = [shk_count]
        failed_jobs = list(job_type_wise_shk_count['failed_job_csv'].keys())
        failed_values = [v[0] for v in job_type_wise_shk_count['failed_job_csv'].values()]
        intact_jobs = list(job_type_wise_shk_count['intact_job_csv'].keys())
        intact_values = [v[0] for v in job_type_wise_shk_count['intact_job_csv'].values()]
        failed_x = range(len(failed_jobs))
        intact_x = range(len(failed_jobs), len(failed_jobs) + len(intact_jobs))
        plt.figure(figsize=(12, 6))
        bars_failed = plt.bar(failed_x, failed_values, color='red', label='Failed Jobs')
        bars_intact = plt.bar(intact_x, intact_values, color='green', label='Success Jobs')
        for bar in list(bars_failed) + list(bars_intact):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height()}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.rcParams.update({'font.size': 15})
        plt.xticks(ticks=list(failed_x) + list(intact_x), labels=failed_jobs + intact_jobs, rotation=75)
        plt.xlabel('Job ID')
        plt.ylabel('Count [Log Scale]')
        plt.yscale('log', base=2)
        plt.title(f'Failed vs. Success Job shock [>{level}(g)] Counts')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


def plot_vib_scatter_motor_turbine(job_wise_cleaned_data):
    bins = [0, 10, 15, 20, np.inf]
    vib_groups = ['0-10', '10-15', '15-20', '>20']
    failed_counts = {g: {'motor': [], 'turbine': []} for g in vib_groups}
    success_counts = {g: {'motor': [], 'turbine': []} for g in vib_groups}
    for job_id, job_data in job_wise_cleaned_data['failed_job_csv'].items():
        counts, _ = np.histogram(job_data['VIB_LAT'], bins=bins)
        key = 'motor' if job_id[-2] == 'M' else 'turbine'
        for i, g in enumerate(vib_groups):
            failed_counts[g][key].append(counts[i])
    for job_id, job_data in job_wise_cleaned_data['intact_job_csv'].items():
        counts, _ = np.histogram(job_data['VIB_LAT'], bins=bins)
        key = 'motor' if job_id[-2] == 'M' else 'turbine'
        for i, g in enumerate(vib_groups):
            success_counts[g][key].append(counts[i])
    plt.figure(figsize=(12, 6))
    for i, group in enumerate(vib_groups):
        for color, label, data in [('red', 'Failed Jobs [motor]', failed_counts[group]['motor']),
                                    ('orange', 'Failed Jobs [turbine]', failed_counts[group]['turbine']),
                                    ('green', 'Success Jobs [motor]', success_counts[group]['motor']),
                                    ('blue', 'Success Jobs [turbine]', success_counts[group]['turbine'])]:
            c = [x if x > 0 else 1 for x in data]
            plt.scatter([group] * len(data), data, color=color,
                        s=np.array(np.log10(c)) * 100, label=label if i == 0 else "")
    plt.xlabel('Lat. Vib. [g]', fontsize=14)
    plt.ylabel('Count [log scale]', fontsize=14)
    plt.title('Vibration Count for Each Job Type by Vibration Range', fontsize=16)
    plt.xticks(rotation=75)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.yscale('log')
    plt.show()


def plot_shock_scatter_motor_turbine(job_wise_cleaned_data):
    bins = [0, 50, 100, 200, 300, 400, 500, np.inf]
    shock_groups = ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500', '>500']
    failed_counts = {g: {'motor': [], 'turbine': []} for g in shock_groups}
    success_counts = {g: {'motor': [], 'turbine': []} for g in shock_groups}
    for job_id, job_data in job_wise_cleaned_data['failed_job_csv'].items():
        counts, _ = np.histogram(job_data['SHK_LAT'], bins=bins)
        key = 'motor' if job_id[-2] == 'M' else 'turbine'
        for i, g in enumerate(shock_groups):
            failed_counts[g][key].append(counts[i])
    for job_id, job_data in job_wise_cleaned_data['intact_job_csv'].items():
        counts, _ = np.histogram(job_data['SHK_LAT'], bins=bins)
        key = 'motor' if job_id[-2] == 'M' else 'turbine'
        for i, g in enumerate(shock_groups):
            success_counts[g][key].append(counts[i])
    plt.figure(figsize=(12, 6))
    for i, group in enumerate(shock_groups):
        for color, label, data in [('red', 'Failed Jobs [motor]', failed_counts[group]['motor']),
                                    ('orange', 'Failed Jobs [turbine]', failed_counts[group]['turbine']),
                                    ('green', 'Success Jobs [motor]', success_counts[group]['motor']),
                                    ('blue', 'Success Jobs [turbine]', success_counts[group]['turbine'])]:
            c = [x if x > 0 else 1 for x in data]
            plt.scatter([group] * len(data), data, color=color,
                        s=np.array(np.log10(c)) * 100, label=label if i == 0 else "")
    plt.xlabel('Lat. Shock [g]', fontsize=14)
    plt.ylabel('Count [log scale]', fontsize=14)
    plt.title('Shock Count for Each Job Type by Shock Range', fontsize=16)
    plt.xticks(rotation=75)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.yscale('log')
    plt.show()


def plot_n2_vib_shk_boxplots(job_ids, job_wise_cleaned_data):
    N2_data = {jt: [] for jt in job_categories}
    vib_data = {jt: [] for jt in job_categories}
    shk_data = {jt: [] for jt in job_categories}
    for job_type in job_categories:
        for job_id in job_ids.get(job_type):
            n2 = [x for x in job_wise_cleaned_data[job_type][job_id]['N2_RATE'].tolist() if x >= 0]
            vib = [x for x in job_wise_cleaned_data[job_type][job_id]['VIB_LAT'].tolist() if 0 < x < 100]
            shk = [x for x in job_wise_cleaned_data[job_type][job_id]['SHK_LAT'].tolist() if 0 < x < 700]
            N2_data[job_type].append(n2)
            vib_data[job_type].append(vib)
            shk_data[job_type].append(shk)
    for job_type in job_categories:
        color = 'red' if job_type == 'failed_job_csv' else 'blue'
        alt_color = 'blue' if job_type == 'failed_job_csv' else 'red'
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(10, 5))
        plt.boxplot(N2_data[job_type], patch_artist=True, labels=job_ids[job_type],
                    boxprops=dict(facecolor=color),
                    flierprops=dict(marker='o', color=alt_color),
                    medianprops=dict(color=alt_color, linewidth=2))
        plt.ylabel('N2 Flow Rate')
        plt.grid()
        plt.suptitle(f"Job Types: [{job_type}]", fontsize=15)
        plt.tight_layout()
        plt.xticks(rotation=75)
        plt.xlabel('Job id', fontsize=15)
        plt.ylim([0, 1250])
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.boxplot(shk_data[job_type], patch_artist=True, labels=job_ids[job_type],
                    boxprops=dict(facecolor=color),
                    flierprops=dict(marker='o', color=alt_color),
                    medianprops=dict(color=alt_color, linewidth=2))
        plt.ylabel('Lat. Vib. [g]')
        plt.grid()
        plt.suptitle(f"Job Types: [{job_type}]", fontsize=15)
        plt.tight_layout()
        plt.xticks(rotation=75)
        plt.xlabel('Job id', fontsize=15)
        plt.show()

if __name__ == "__main__":
    print("This module contains functions for plotting and is not meant to be run directly.")
    pass
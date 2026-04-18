import sys
sys.path.append('..')

directory_intact_jobs = "*/DataWOFail/AllJobsData"
directory_failed_jobs = "*/Data/AllJobsData"

job_ids_motor_turbine = {
    'motor': ['O.1048592.41-7','O.1048592.59-6','O.1048592.107-3','O.1048592.107-6',
              'O.1048592.107-7','O.1048592.107-9','O.1048592.107-10','O.1048592.133-1'],
    'turbine': ['O.1048592.72-18','O.1048592.99-5','O.1048592.59-9','O.1048592.59-10',
                'O.1048592.110-2','O.1048592.110-4','O.1048592.110-5','O.1048592.110-6',
                'O.1048592.133-5']
}

job_categories = ["failed_job_csv", "intact_job_csv"]

activities = ['On Bottom Drilling', 'Pull Test', 'Wiper Trip', 'Trip In Run', 'Trip Out Run']

columns_of_interest = ['TIME','BVEL','CT_WGT','DEPT','HDTH','FLWI','APRS_RAW','IPRS_RAW','N2_RATE','VIB_LAT','SHK_LAT']

if __name__ == "__main__":
    print("This module contains configuration variables and is not meant to be run directly.")
    pass

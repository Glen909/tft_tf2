# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gc
import glob
import os
import shutil
import sys

from expt_settings.configs import ExperimentConfig
import numpy as np
import pandas as pd
import pyunpack
import wget
from py7zr import unpack_7zarchive
import datetime


def recreate_folder(path):
  """Deletes and recreates folder."""

  shutil.rmtree(path)
  os.makedirs(path)

def process_target_usi(config):

  data_folder = config.data_folder

  # 记得把数据放到 data/target_usi
  print('Regenerating data...')

  # load multiple tripdata
  # merging the files
  joined_files = os.path.join(os.path.join(data_folder), "yellow_tripdata_*.csv")

  # A list of all joined files is returned
  joined_list = glob.glob(joined_files)

  # # Finally, the files are joined
  tripdata_merged = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
  tripdata_merged.insert(0, 'index', range(len(df)), allow_duplicates=False)
  # merged
  print('tripdata merged')

  holidays = pd.read_csv(os.path.join(data_folder, 'holidays.csv'))
  merged_TreeCover = pd.read_csv(os.path.join(data_folder, 'merged_TreeCover.csv'))
  weather = pd.read_csv(os.path.join(data_folder, 'weather.csv'))
  # load done
  print('load all data done')

  # Add trajectory identifier
  tripdata_merged['traj_id'] = tripdata_merged['PULocationID'].apply(
      str) + '_' + tripdata_merged['DOLocationID'].apply(str)
  tripdata_merged['unique_id'] = tripdata_merged['traj_id'] + '_' + tripdata_merged['tpep_dropoff_datetime'].apply(
      str)
  
  # Adding holidays
  print('Adding holidays')
  # drop_hms_date 用来判断节假日 
  tripdata_merged['drop_hms_date'] = tripdata_merged['tpep_dropoff_datetime'].dt.strftime('%Y-%m-%d')
  tripdata_merged = tripdata_merged.merge(
      holidays,
      on='drop_hms_date',
      how='left')
  # 非假期的日期用-1来填充
  tripdata_merged['holiday_is_True'] = tripdata_merged['holiday_is_True'].fillna(-1)
  print('Adding holidays Done')

  print('Adding weather')
  #首先提取日期，温度，湿度，风速，能见度，太阳辐射指数
  mini_weather = pd.DataFrame(weather[[
    'drop_hms_date',
    'temp',
    'humidity',
    'windspeed',
    'visibility',
    'solarradiation'
    ]])
  tripdata_merged = tripdata_merged.join(mini_weather, on='drop_hms_date', how='left')
  print('Adding weather Done')

  # Adding treecover
  print('Adding treecover')
  merged_TreeCover['TreeCover'].fillna(method='ffill', inplace=True)
  tripdata_merged = tripdata_merged.join(merged_TreeCover, on='DOLocationID', how='left')

  # Additional date info
  tripdata_merged['day_of_week'] = pd.to_datetime(tripdata_merged['drop_hms_date'].values).dayofweek
  tripdata_merged['day_of_month'] = pd.to_datetime(tripdata_merged['drop_hms_date'].values).day
  tripdata_merged['month'] = pd.to_datetime(tripdata_merged['drop_hms_date'].values).month

  
  tripdata_merged.sort_values('unique_id', inplace=True)

  print('Saving processed file to {}'.format(config.data_csv_path))
  tripdata_merged.to_csv(config.data_csv_path)


# Core routine.
def main(expt_name, output_folder):
  """Runs main download routine.

  Args:
    expt_name: Name of experiment
    output_folder: Folder path for storing data
  """

  print('#### Running process target usi data script ###')

  expt_config = ExperimentConfig(expt_name, output_folder)

  if os.path.exists(expt_config.data_csv_path):
    print('Data has been processed for {}. Skipping download...'.format(
        expt_name))
    sys.exit(0)
  else:
    print('Resetting data folder...')
    recreate_folder(expt_config.data_folder)

  # Default download functions
  download_functions = {
      'volatility': download_volatility,
      'electricity': download_electricity,
      'traffic': download_traffic,
      'favorita': process_favorita,
      'target_usi': process_target_usi
  }

  if expt_name not in download_functions:
    raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

  download_function = download_functions[expt_name]

  # Run data download
  print('Getting {} data...'.format(expt_name))
  download_function(expt_config)

  print('Download completed.')


if __name__ == '__main__':

  def get_args():
    """Returns settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description='Data download configs')
    parser.add_argument(
        'expt_name',
        metavar='e',
        type=str,
        nargs='?',
        choices=experiment_names,
        help='Experiment Name. Default={}'.format(','.join(experiment_names)))
    parser.add_argument(
        'output_folder',
        metavar='f',
        type=str,
        nargs='?',
        default='.',
        help='Path to folder for data download')

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == '.' else args.output_folder

    return args.expt_name, root_folder

  name, folder = get_args()
  main(expt_name=name, output_folder=folder)

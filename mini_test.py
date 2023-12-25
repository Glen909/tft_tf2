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

import numpy as np
import pandas as pd
from datetime import datetime


def process_target_usi():


  # 记得把数据放到 data/target_usi
  print('Regenerating data...')
  print('begin to merge all yellow data')
  dtype = {
    'VendorID': str, # int8
    # 'tpep_pickup_datetime': object, #保留
    # 'tpep_dropoff_datetime': object,#保留
    'passenger_count': str,#保留 int8
    'trip_distance': 'float32',#保留
    'RatecodeID': str, # int8
    'store_and_fwd_flag': str,
    'PULocationID': 'int16',#保留
    'DOLocationID': 'int16',#保留
    'payment_type': str, # int8
    'fare_amount': 'float32',#保留
    'extra': 'float32',#保留
    'mta_tax': 'float32',#保留
    'tip_amount': str,
    'tolls_amount': str,
    'improvement_surcharge': str,
    'total_amount': 'float32',#保留
    'congestion_surcharge': 'float32', #保留
  }

  joined_files = os.path.join('/home/featurize/data', "yellow_*.csv")

  joined_list = glob.glob(joined_files)

  df_list = [pd.read_csv(file, parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'], dtype=dtype) for file in joined_list]

  tripdata_merged = pd.concat(df_list, ignore_index=True)

  del tripdata_merged['VendorID']
  del tripdata_merged['RatecodeID']
  del tripdata_merged['store_and_fwd_flag']
  del tripdata_merged['improvement_surcharge']
  del tripdata_merged['tolls_amount']
  del tripdata_merged['payment_type']
  del tripdata_merged['tip_amount']

  #tripdata_merged.fillna(tripdata_merged.mean(), inplace=True)
  tripdata_merged['passenger_count'] = tripdata_merged['passenger_count'].astype('Int8')
  tripdata_merged['passenger_count'].fillna(tripdata_merged['passenger_count'].median(), inplace=True)
  tripdata_merged['congestion_surcharge'].fillna(tripdata_merged['congestion_surcharge'].mean(), inplace=True)

  gc.collect()
  print('all yellow data merge done')
  #print(tripdata_merged.dtypes)
  
  print('begin to load holidays')
  holidays = pd.read_csv(os.path.join('/home/featurize/data', 'holidays.csv'))

  tripdata_merged['drop_hms_date'] = tripdata_merged['tpep_dropoff_datetime'].dt.strftime('%Y-%m-%d')
  tripdata_merged = tripdata_merged.merge(
      holidays,
      on='drop_hms_date',
      how='left')
  tripdata_merged['holiday_is_True'].fillna(value=0, inplace=True)
  #tripdata_merged.set_index('drop_hms_date', inplace=True)
  del holidays
  gc.collect()
  print('Adding holidays Done')

  print('begin to add treecover')
  dtype_tree = {
    'DOLocationID': int,
    'Borough': str,
    'Zone': str,
    'service_zone': str,
    'TreeCover': float,
  }
  merged_TreeCover = pd.read_csv(os.path.join('/home/featurize/data', 'merged_TreeCover.csv'), dtype=dtype_tree)
  mini_merged_TreeCover = pd.DataFrame(merged_TreeCover[[
    'DOLocationID',
    'TreeCover'
  ]])
  del merged_TreeCover
  tripdata_merged = tripdata_merged.merge(mini_merged_TreeCover, on='DOLocationID', how='left')
  tripdata_merged['TreeCover'].fillna(tripdata_merged['TreeCover'].mean(), inplace=True)
  tripdata_merged['TreeCover'] = tripdata_merged['TreeCover'].round(2)
  del mini_merged_TreeCover
  gc.collect()
  print('adding treecover done')

  print('begin to Add weather')
  weather = pd.read_csv(os.path.join('/home/featurize/data', 'weather.csv'))
  #首先提取日期，温度，湿度，风速，能见度，太阳辐射指数
  mini_weather = pd.DataFrame(weather[[
    'drop_hms_date',
    'temp',
    'humidity',
    'windspeed',
    'visibility',
    'solarradiation'
    ]])
  del weather
  tripdata_merged = tripdata_merged.merge(mini_weather, on='drop_hms_date', how='left')
  tripdata_merged['temp'].ffill(inplace=True)
  tripdata_merged['humidity'].ffill(inplace=True)
  tripdata_merged['windspeed'].ffill(inplace=True)
  tripdata_merged['visibility'].ffill(inplace=True)
  tripdata_merged['solarradiation'].ffill(inplace=True)
  gc.collect()
  print('Adding weather Done')

  print('additional information')
  tripdata_merged['drop_hms_date'] = pd.to_datetime(tripdata_merged['drop_hms_date'])

  # 提取各种时间信息并存储在新列中
  tripdata_merged['day_of_week'] = tripdata_merged['drop_hms_date'].dt.dayofweek.astype('uint8')
  tripdata_merged['day_of_month'] = tripdata_merged['drop_hms_date'].dt.day.astype('uint8')
  tripdata_merged['month'] = tripdata_merged['drop_hms_date'].dt.month.astype('uint8')
  
  # Add trajectory identifier
  tripdata_merged['traj_id'] = tripdata_merged['month'].apply(str
    ) + '_' + tripdata_merged['day_of_month'].apply(str
    ) + '_' + tripdata_merged['PULocationID'].apply(str
    ) + '_' + tripdata_merged['DOLocationID'].apply(str)
  print('traj_id done')
  tripdata_merged['unique_id'] = tripdata_merged['traj_id'] + '_' + tripdata_merged['tpep_dropoff_datetime'].apply(str)
  print('unique_id done')
  tripdata_merged.sort_values('unique_id', inplace=True)
  print('Saving processed file to ')
  print(tripdata_merged.isnull().any())
  tripdata_merged.to_csv('miniout.csv', index=False)
  print("done done done")
  # print(tripdata_merged.isnull().any())
  # print(tripdata_merged[tripdata_merged['passenger_count'].isnull()])



if __name__ == '__main__':
  print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  process_target_usi()
  print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

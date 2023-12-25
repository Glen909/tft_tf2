# coding=utf-8
"""Custom formatting functions for Favorita dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class TargetUsiFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the Favorita dataset.  

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """
  

  _column_definition = [
      ('traj_id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('drop_hms_date', DataTypes.DATE, InputTypes.TIME),
      ('DOLocationID', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('passenger_count', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('trip_distance', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('PULocationID', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('fare_amount', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('extra', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('mta_tax', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('total_amount', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('congestion_surcharge', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('holiday_is_True', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('TreeCover', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('temp', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('humidity', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('windspeed', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('visibility', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('solarradiation', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df, valid_boundary=None, test_boundary=None):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    if valid_boundary is None:
      valid_boundary = pd.datetime(2019, 12, 1)

    fixed_params = self.get_fixed_params()
    time_steps = fixed_params['total_time_steps']
    lookback = fixed_params['num_encoder_steps']
    forecast_horizon = time_steps - lookback

    df['drop_hms_date'] = pd.to_datetime(df['drop_hms_date'])
    df_lists = {'train': [], 'valid': [], 'test': []}
    for _, sliced in df.groupby('traj_id'):
      index = sliced['drop_hms_date']
      train = sliced.loc[index < valid_boundary]
      train_len = len(train)
      valid_len = train_len + forecast_horizon
      valid = sliced.iloc[train_len - lookback:valid_len, :]
      test = sliced.iloc[valid_len - lookback:valid_len + forecast_horizon, :]

      sliced_map = {'train': train, 'valid': valid, 'test': test}

      for k in sliced_map:
        item = sliced_map[k]

        if len(item) >= time_steps:
          df_lists[k].append(item)

    dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

    train = dfs['train']
    self.set_scalers(train, set_real=True)

    # Use all data for label encoding  to handle labels not present in training.
    self.set_scalers(df, set_real=False)

    # Filter out identifiers not present in training (i.e. cold-started items).
    def filter_ids(frame):
      identifiers = set(self.identifiers)
      index = frame['traj_id']
      return frame.loc[index.apply(lambda x: x in identifiers)]

    valid = filter_ids(dfs['valid'])
    test = filter_ids(dfs['test'])

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df, set_real=True):
    """Calibrates scalers using the data supplied.

    Label encoding is applied to the entire dataset (i.e. including test),
    so that unseen labels can be handled at run-time.

    Args:
      df: Data to use to calibrate scalers.
      set_real: Whether to fit set real-valued or categorical scalers
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    if set_real:

      # Extract identifiers in case required
      self.identifiers = list(df[id_column].unique())

      # Format real scalers
      self._real_scalers = {}
      for col in ['oil', 'transactions', 'log_sales']:
        self._real_scalers[col] = (df[col].mean(), df[col].std())

      self._target_scaler = (df[target_column].mean(), df[target_column].std())

    else:
      # Format categorical scalers
      categorical_inputs = utils.extract_cols_from_data_type(
          DataTypes.CATEGORICAL, column_definitions,
          {InputTypes.ID, InputTypes.TIME})

      categorical_scalers = {}
      num_classes = []
      if self.identifiers is None:
        raise ValueError('Scale real-valued inputs first!')
      id_set = set(self.identifiers)
      valid_idx = df['traj_id'].apply(lambda x: x in id_set)
      for col in categorical_inputs:
        # Set all to str so that we don't have mixed integer/string columns
        srs = df[col].apply(str).loc[valid_idx]
        categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
            srs.values)

        num_classes.append(srs.nunique())

      # Set categorical scaler outputs
      self._cat_scalers = categorical_scalers
      self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    for col in ["DOLocationID", "passenger_count", "trip_distance", "PULocationID", "fare_amount", "extra", "mta_tax", "total_amount", "congestion_surcharge", "TreeCover", "temp", "humidity", "windspeed", "visibility", "solarradiation", "day_of_month", "month"]:
      mean, std = self._real_scalers[col]
      output[col] = (df[col] - mean) / std

      if col == 'log_sales':
        output[col] = output[col].fillna(0.)  # mean imputation

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns
    mean, std = self._target_scaler
    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = (predictions[col] * std) + mean

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 120,
        'num_encoder_steps': 90,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 240,
        'learning_rate': 0.001,
        'minibatch_size': 128,
        'max_gradient_norm': 100.,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return 450000, 50000

  def get_column_definition(self):
    """"Formats column definition in order expected by the TFT.

    Modified for Favorita to match column order of original experiment.

    Returns:
      Favorita-specific column definition
    """

    column_definition = self._column_definition

    # Sanity checks first.
    # Ensure only one ID and time column exist
    def _check_single_column(input_type):

      length = len([tup for tup in column_definition if tup[2] == input_type])

      if length != 1:
        raise ValueError('Illegal number of inputs ({}) of type {}'.format(
            length, input_type))

    _check_single_column(InputTypes.ID)
    _check_single_column(InputTypes.TIME)

    identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
    time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
    real_inputs = [
        tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
        tup[2] not in {InputTypes.ID, InputTypes.TIME}
    ]

    col_definition_map = {tup[0]: tup for tup in column_definition}
    col_order = [
        "tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count",
        "trip_distance","PULocationID","DOLocationID","fare_amount","extra",
        "mta_tax","total_amount","congestion_surcharge","drop_hms_date",
        "holiday_is_True","TreeCover","temp","humidity","windspeed",
        "visibility","solarradiation","day_of_week","day_of_month","month",
        "traj_id","unique_id"
    ]
    categorical_inputs = [
        col_definition_map[k] for k in col_order if k in col_definition_map
    ]

    return identifier + time + real_inputs + categorical_inputs

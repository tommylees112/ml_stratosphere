import numpy as np
import calendar
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
import pickle
import xarray as xr
import warnings

from typing import cast, DefaultDict, Dict, List, Optional, Union, Tuple


class StratoEngineer:
    name: str = 'strato'

    relevant_months: List[List[int]] = [[10, 11, 12], [1, 2, 3]]

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder

        self.interim_folder = data_folder / 'interim'
        assert self.interim_folder.exists(), \
            f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        try:
            # specific folder for that
            self.output_folder = data_folder / 'features' / self.name
            if not self.output_folder.exists():
                self.output_folder.mkdir(parents=True)
        except AttributeError:
            print('Name not defined! No experiment folder set up')

    def engineer(self, test_year: Union[int, List[int]],
                 target_variable: str = 'VHI',
                 pred_days: int = 12,
                 expected_length: Optional[int] = 12,
                 ) -> None:
        """
        Note in this case, pred months is pred_days
        TODO: rename pred_months to pred_units, and make the month an attribute of the class
        """

        self._process_dynamic(test_year, target_variable, pred_days, expected_length)

    def _process_dynamic(self, test_year: Union[int, List[int]],
                         target_variable: str = 'VHI',
                         pred_months: int = 12,
                         expected_length: Optional[int] = 12,
                         ) -> None:
        if expected_length is None:
            warnings.warn('** `expected_length` is None. This means that \
            missing data will not be skipped. Are you sure? **')

        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset()

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        train_ds = self._train_test_split(
            ds=data, years=cast(List, test_year),
            target_variable=target_variable, pred_days=pred_months,
            expected_length=expected_length,
        )

        # normalization_values = self._calculate_normalization_values(train_ds)

        # split train_ds into x, y for each year-month before `test_year` & save
        self._stratify_training_data(
            train_ds=train_ds, target_variable=target_variable,
            pred_months=pred_months
        )

        # savepath = self.output_folder / 'normalizing_dict.pkl'
        # with savepath.open('wb') as f:
        #     pickle.dump(normalization_values, f)

    def _get_preprocessed_files(self) -> List[Path]:
        processed_files = []
        interim_folder = self.interim_folder
        for subfolder in interim_folder.iterdir():
            if str(subfolder).endswith('_preprocessed') and subfolder.is_dir():
                processed_files.extend(list(subfolder.glob('*.nc')))
        return processed_files

    def _make_dataset(self) -> xr.Dataset:

        datasets = []
        for idx, file in enumerate(self._get_preprocessed_files()):
            print(f'Processing {file}')
            datasets.append(xr.open_dataset(file))

        # join all preprocessed datasets
        main_dataset = datasets[0]
        for dataset in datasets[1:]:
            # ensure equal timesteps ('inner' join)
            main_dataset = main_dataset.merge(dataset, join='inner')

        return main_dataset

    def _stratify_training_data(self, train_ds: xr.Dataset,
                                target_variable: str,
                                pred_months: int
                                ) -> None:
        """split `train_ds` into x, y and save the outputs to
        self.output_folder (data/features) """

        min_date = self._get_datetime(train_ds.time.values.min())
        min_year, min_month = min_date.year, min_date.month
        max_date = self._get_datetime(train_ds.time.values.max())
        max_year, max_month = max_date.year, max_date.month

        cur_pred_year, cur_pred_month = max_date.year, max_date.month

        group_year_month: List[List[Tuple[int, int]]] = []
        for year in range(min_year, max_year - 1):
            group: List[Tuple[int, int]] = []
            for months in self.relevant_months[0]:
                for month in months:
                    group.append((year, month))
            for months in self.relevant_months[1]:
                for month in months:
                    group.append((year + 1, month))
            group_year_month.append(group)

        for group in group_year_month:
            for year, month in group:
                if date(year, month, 1) < max_date:
                    arrays, _, triggered = self._stratify_xy(
                        ds=train_ds, year=cur_pred_year,
                        target_variable=target_variable, target_month=cur_pred_month,
                        pred_days=pred_months,
                        )
                    self._save(
                        arrays, year=cur_pred_year, month=cur_pred_month,
                        dataset_type='train'
                    )
                    if triggered:
                        break

    def _train_test_split(self, ds: xr.Dataset,
                          years: List[int],
                          target_variable: str,
                          pred_days: int,
                          expected_length: Optional[int]
                          ) -> xr.Dataset:
        """save the test data and return the training dataset"""

        years.sort()

        # for the first `year` Jan calculate the xy_test dictionary and min date
        group_year_month: List[List[Tuple[int, int]]] = []
        group: List[Tuple[int, int]] = []
        for year in years:
            for month in self.relevant_months[0]:
                group.append((year, month))
            for month in self.relevant_months[1]:
                group.append((year + 1, month))
            group_year_month.append(group)
            group = []

        xy_tests, min_test_date, triggered = self._stratify_xy(
            ds=ds, year=group_year_month[0][0][0], target_variable=target_variable,
            target_month=group_year_month[0][0][1], pred_days=pred_days)

        # the train_ds MUST BE from before minimum test date
        train_dates = ds.time.values <= np.datetime64(str(min_test_date))
        train_ds = ds.isel(time=train_dates)

        # save the xy_test dictionary
        self._save(
            xy_tests, year=group_year_month[0][0][0], month=group_year_month[0][0][1],
            dataset_type='test'
        )

        # each month in test_year produce an x,y pair for testing
        for group in group_year_month:
            for year, month in group:
                # prevents the initial test set from being recalculated
                xy_test, _, triggered = self._stratify_xy(
                    ds=ds, year=year, target_variable=target_variable,
                    target_month=month, pred_days=pred_days
                )
                if xy_test is not None:
                    self._save(
                        xy_test, year=year, month=month,
                        dataset_type='test'
                    )
                if triggered:
                    break

        return train_ds

    def _stratify_week(self, ds: xr.Dataset, target_variable: str,
                       min_date: np.datetime64,
                       pred_days: int) -> Tuple[Dict[str, xr.Dataset], bool]:
        max_date = min_date + timedelta(days=7)

        # `max_date` is the date to be predicted;
        # `max_train_date` is one timestep before;
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))

        # lets also get the training data stuff
        min_train_date = min_date - timedelta(days=pred_days)
        min_train_date_np = np.datetime64(str(min_train_date))

        x = (ds.time.values > min_train_date_np) & (ds.time.values <= min_date_np)
        y = (ds.time.values > min_date_np) & (ds.time.values <= max_date_np)

        x_dataset = ds.isel(time=x)
        y_dataset = ds.isel(time=y)[target_variable].to_dataset(name=target_variable)

        # right now, y is a netcdf file of a week's worth of data. We want to turn it into a boolean
        # assuming if it dips below 0, we have found the incident we are looking for
        triggered = False
        if (y_dataset[target_variable] < 0).any():
            y_dataset[target_variable] = 1
            triggered = True
        else:
            y_dataset[target_variable] = 0

        return {'x': x_dataset, 'y': y_dataset}, triggered

    def _stratify_xy(self, ds: xr.Dataset,
                     year: int,
                     target_variable: str,
                     target_month: int,
                     pred_days: int,
                     ) -> Tuple[List[Dict[str, xr.Dataset]], date, bool]:

        """
        We are trying to predict: will an event happen in the next week?
        """

        print(f"Generating data for year: {year}, target month: {target_month}")

        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        min_date = date(year, target_month, 1)

        # for each month, we will take 4 weeks as "test" data
        test_xys = []
        for i in range(4):
            min_date = min_date + timedelta(days=i * 7)  # 7 days in a week
            test_xy, triggered = self._stratify_week(ds, target_variable, min_date, pred_days)
            test_xys.append(test_xy)

            if triggered:
                break

        print(
            f"Max date: {str(max_date)}, "
            f"min input date: {str(min_date)}"
        )

        return test_xys, date(year, target_month, 1), triggered

    @staticmethod
    def _get_datetime(time: np.datetime64) -> date:
        return datetime.strptime(time.astype(str)[:10], '%Y-%m-%d').date()

    def _save(self, ds_dicts: List[Dict[str, xr.Dataset]], year: int,
              month: int, dataset_type: str) -> None:

        save_folder = self.output_folder / dataset_type
        save_folder.mkdir(exist_ok=True)

        for idx, week in enumerate(ds_dicts):
            output_location = save_folder / f'{year}_{month}_{idx + 1}'
            output_location.mkdir(exist_ok=True)
            for x_or_y, output_ds in week.items():
                print(f'Saving data to {output_location.as_posix()}/{x_or_y}.nc')
                output_ds.to_netcdf(output_location / f'{x_or_y}.nc')

    def _calculate_normalization_values(self,
                                        x_data: xr.Dataset) -> DefaultDict[str, Dict[str, float]]:
        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in x_data.data_vars:
            mean = float(x_data[var].mean(dim=['lat', 'lon', 'time'], skipna=True).values)
            std = float(x_data[var].std(dim=['lat', 'lon', 'time'], skipna=True).values)
            normalization_values[var]['mean'] = mean
            normalization_values[var]['std'] = std

        return normalization_values

    @staticmethod
    def _make_fill_value_dataset(ds: Union[xr.Dataset, xr.DataArray],
                                 fill_value: Union[int, float] = -9999.0,
                                 ) -> Union[xr.Dataset, xr.DataArray]:
        nan_ds = xr.full_like(ds, fill_value)
        return nan_ds

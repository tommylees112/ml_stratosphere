from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from pathlib import Path
import pickle

from typing import Dict, List, Tuple, Optional

from ..models.data import DataLoader
from ..models.neural_networks.base import NNBase


int2month = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}


def plot_shap_values(x: np.ndarray,
                     shap_values: np.ndarray,
                     val_list: List[str],
                     normalizing_dict: Dict[str, Dict[str, float]],
                     value_to_plot: str,
                     normalize_shap_plots: bool = True,
                     show: bool = False,
                     polished_value_name: Optional[str] = None,
                     pred_date: Optional[Tuple[int, int]] = None,
                     fig: Optional[Figure] = None) -> None:
    """Plots the denormalized values against their shap values, so that
    variations in the input features to the model can be compared to their effect
    on the model. For example plots, see notebooks/08_gt_recurrent_model.ipynb.
    Parameters:
    ----------
    x: np.array
        The input to a model for a single data instance
    shap_values: np.array
        The corresponding shap values (to x)
    val_list: list
        A list of the variable names, for axis labels
    normalizing_dict: dict
        The normalizing dict saved by the `Engineer`, so that the x array can be
        denormalized
    value_to_plot: str
        The specific input variable to plot. Must be in val_list
    normalize_shap_plots: bool = True
        If True, then the scale of the shap plots will be uniform across all
        variable plots (on an instance specific basis).
    show: bool = False
        If True, a plot of the variable `value_to_plot` against its shap values will be plotted.
    polished_value_name: Optional[str] = None
        If passed to the model, this is used instead of value_to_plot when labelling the axes.
    pred_month: Optional[Tuple[int, int]] = None
        If passed to the model, the x axis will contain actual months instead of the index.
        Note the tuple is [int_month, int_year]
    fig: Optional[Figure] = None
        The figure upon which to construct the plot. If None is passed, matplotlib will use
        plt.gcf() to get the figure
    """
    # first, lets isolate the lists
    idx = val_list.index(value_to_plot)

    x_val = x[:, idx]

    # we also want to denormalize
    for norm_var in normalizing_dict.keys():
        if value_to_plot.endswith(norm_var):
            x_val = (x_val * normalizing_dict[norm_var]['std']) + \
                normalizing_dict[norm_var]['mean']
            break

    shap_val = shap_values[:, idx]

    months = list(range(1, len(x_val) + 1))

    if pred_date is not None:
        int_months, int_years = [], []
        cur_month, cur_year = pred_date[0], pred_date[1]
        for i in range(1, len(x_val) + 1):
            cur_month = cur_month - 1
            if cur_month == 0:
                cur_month = 12
                cur_year -= 1
            int_months.append(cur_month)
            int_years.append(cur_year)
        str_dates = [f'{int2month[m]}{y}' for m, y in zip(int_months, int_years)][::-1]

    host = host_subplot(111, axes_class=AA.Axes, figure=fig)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par1.axis["right"].toggle(all=True)

    if normalize_shap_plots:
        par1.set_ylim(shap_values.min(), shap_values.max())

    if polished_value_name is None:
        polished_value_name = value_to_plot

    host.set_xlabel("Months")
    host.set_ylabel(polished_value_name)
    par1.set_ylabel("Shap value")

    p1, = host.plot(months, x_val, label=polished_value_name, linestyle='dashed')
    p2, = par1.plot(months, shap_val, label="Shap value")

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    host.legend(loc='lower left', framealpha=0.5)

    if pred_date is not None:
        modulo = (len(months) - 1) % 2
        host.set_xticks(months[modulo::2])
        host.set_xticklabels(str_dates[modulo::2])

    plt.draw()
    if show:
        plt.show()


def all_shap_for_file(test_folder: Path,
                      model: NNBase,
                      background_size: int = 100,
                      batch_size: int = 100) -> None:
    """
    Calculate all the shap values for a single file (i.e. for all the
    data instances in that file).

    The calculated shap values are saved in the model's analysis folder
    (i.e. model_dir / 'analysis').

    Warning: this function can take quite a while to run.

    Arguments:
    ----------
    test_file: Path
        A Path to the test folder containing the test file. This assumes the test
        folder was generated by the pipeline's engineering class.
    model: NNBase
        A neural network model object, as defined by the pipeline (i.e. an EALSTM, RNN
        or linear network)
    background: int = 100
        The number of background training samples to use
    batch_size: int = 100
        The size of the batches to use when calculating shap values. If you are getting memory
        errors, reducing this is a good place to start
    """
    static = model.include_static
    monthly_aggs = model.include_monthly_aggs
    ignore_vars = model.ignore_vars
    surrounding_pixels = model.surrounding_pixels
    experiment = model.experiment

    data_path = test_folder.parents[3]

    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test', to_tensor=True,
                                    static=static, experiment=experiment,
                                    surrounding_pixels=surrounding_pixels,
                                    ignore_vars=ignore_vars,
                                    monthly_aggs=monthly_aggs)

    test_arrays_loader.data_files = [test_folder]

    key, val = list(next(iter(test_arrays_loader)).items())[0]

    output_dict: Dict[str, np.ndarray] = {}

    num_inputs = val.x.historical.shape[0]
    print(f'Calculating shap values for {num_inputs} instances')
    start_idx = 0

    while start_idx < num_inputs:
        print(f'Calculating shap values for indices {start_idx} to {start_idx + batch_size}')
        var_names = None
        if start_idx == 0:
            var_names = val.x_vars
        shap_inputs = model.make_shap_input(val.x, start_idx=start_idx,
                                            num_inputs=batch_size)
        explanations = model.explain(x=shap_inputs, var_names=var_names,
                                     save_shap_values=False, background_size=background_size)

        if start_idx == 0:
            for input_name, shap_array in explanations.items():
                output_dict[input_name] = shap_array
        else:
            for input_name, shap_array in explanations.items():
                output_dict[input_name] = np.concatenate((output_dict[input_name],
                                                          shap_array),
                                                         axis=0)
        start_idx = start_idx + batch_size

    print('Saving results')

    analysis_folder = model.model_dir / 'analysis'

    # this assumes the test file was taken from the data directory, in which case the
    # folder its in is a year_month identifier
    file_id = test_folder.parts[-1]
    file_id_folder = analysis_folder / file_id

    if not file_id_folder.exists():
        file_id_folder.mkdir(parents=True)

    for output_type, shap_array in output_dict.items():
        np.save(file_id_folder / f'shap_value_{output_type}.npy', shap_array)

    with (file_id_folder / 'input_ModelArray.pkl').open('wb') as f:
        pickle.dump(val, f)

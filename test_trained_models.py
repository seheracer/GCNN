import os
import tqdm
import json
import pytest

import torch

from models.PNNStack import PNNStack
from data_utils.raw_dataset_loader import RawDataLoader
from data_utils.serialized_dataset_loader import SerializedDataLoader
from data_utils.dataset_descriptors import (
    AtomFeatures,
    Dataset,
)
from data_utils.dataset_descriptors import AtomFeatures
from utils.utils import dataset_loading_and_splitting, test
from utils.models_setup import generate_model
from utils.visualizer import Visualizer


def best_models(
    model_index,
    models_dir="./reproducing_results/best_performing_models/",
    plot_results=False,
):
    expected_error = [
        0.0011474628477206312,
        4.2980859073136316e-05,
        0.09525396614461332,
    ]
    EPSILON = 1e-3

    available_models = sorted(os.listdir(models_dir))
    try:
        chosen_model = available_models[model_index]
    except IndexError:
        print("model_index must be less than ", len(available_models))

    for m, model in enumerate(available_models):
        print(m, ")", model)

    with open(f"{models_dir}{chosen_model}/config.json", "r") as f:
        config = json.load(f)

    chosen_dataset_option = [x for x in Dataset if x.value == config["dataset_option"]][
        0
    ]
    os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()
    train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        config=config,
        chosen_dataset_option=chosen_dataset_option,
    )
    model = generate_model(
        model_type="PNN",
        input_dim=len(config["atom_features"]),
        dataset=train_loader.dataset,
        config=config,
    )
    state_dict = torch.load(
        f"./{models_dir}{chosen_model}/{chosen_model}.pk",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)

    error, true_values, predicted_values = test(
        test_loader, model, config["output_dim"]
    )
    assert abs(error - expected_error[model_index]) < EPSILON

    if plot_results:
        visualizer = Visualizer("")
        visualizer.add_test_values(
            true_values=true_values, predicted_values=predicted_values
        )
        visualizer.create_scatter_plot(save_plot=False)


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("model_index", [0, 1, 2])
def pytest_best_models(model_index):
    best_models(model_index)


if __name__ == "__main__":
    model_index = int(
        input(
            "Type the number of the model for which you want to reproduce the results with (0, 1, or 2): "
        )
    )
    best_models(model_index, plot_results=True)

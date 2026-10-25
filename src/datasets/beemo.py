from src.datasets.perich_miller import PerichMillerDataset
from src.datasets.nested import NestedSpikingDataset
from src.datasets.churchland import ChurchlandDataset
from src.datasets.gallego import GallegoDataset
from src.datasets.ma_miller import MaMillerDataset


def BeemoDataset(root, transform=None, dataset_list=None, **kwargs):
    if dataset_list is None:
        dataset_list = ["mp", "churchland", "gallego", "miller"]

    # Map dataset names to their constructors
    dataset_map = {
        "mp": PerichMillerDataset(root, **kwargs) if "mp" in dataset_list else None,
        "churchland": (
            ChurchlandDataset(root, **kwargs) if "churchland" in dataset_list else None
        ),
        "gallego": (
            GallegoDataset(root, **kwargs) if "gallego" in dataset_list else None
        ),
        "miller": (
            MaMillerDataset(root, **kwargs) if "miller" in dataset_list else None
        ),
    }

    ds_dict = {}

    for dataset in dataset_list:
        if dataset not in dataset_map:
            raise ValueError(f"Invalid dataset: {dataset}")
        if dataset_map[dataset] is not None:
            ds_dict[dataset] = dataset_map[dataset]

    ds = NestedSpikingDataset(
        datasets={
            **ds_dict,
        },
        transform=transform,
    )
    return ds

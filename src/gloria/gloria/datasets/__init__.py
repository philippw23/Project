from . import data_module
from . import pretraining_dataset
from . import bonetumor_dataset

DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,
    "bonetumor": data_module.BoneTumorDataModule,
}

from llmtuner import run_exp
from utils.set_dataset_path import update_dataset_config

def main():
    update_dataset_config()
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

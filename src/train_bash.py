from llmtuner import run_exp

import wandb
wandb.init(
    name="Baichuan2-13B-Base-Sfted-Mixed-PPO-V1",
    project="huggingface"
)

def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

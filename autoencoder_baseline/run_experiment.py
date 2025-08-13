from autoencoder_baseline.utils.config import Config
from autoencoder_baseline.utils.training import AutoEncoderTrainer
from autoencoder_baseline.utils.visualisation import visualise_combined


if __name__ == "__main__":
    config_path = (
        "/g/kreshuk/tempus/projects/autoencoder_experiments/000_test/config.yaml"
    )
    config = Config.from_yaml(config_path)

    trainer = AutoEncoderTrainer(config.model, config.training, config.paths)

    trainer.train()

    if config.paths.figs_dir is not None:
        visualise_combined(
            config.paths.checkpoints_dir,
            config.paths.figs_dir / (config.experiment_name + "_imgs.png"),
        )

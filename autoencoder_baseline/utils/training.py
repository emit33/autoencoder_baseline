from dataclasses import asdict
import os
import torch
from tqdm import tqdm

from autoencoder_baseline.utils.data_handling import (
    get_train_dataloader,
)
from autoencoder_baseline.utils.autoencoder import AutoEncoder
from autoencoder_baseline.utils.config import ModelConfig, TrainingConfig, PathConfig


class AutoEncoderTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        path_config: PathConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.path_config = path_config

        # Store states
        self.n_epochs = training_config.n_epochs
        self.save_ckpt_step = training_config.save_ckpt_step

        self.checkpoint_dir = self.path_config.checkpoints_dir

        self.device = training_config.device

        # Get dataloaders
        self.dataloader = get_train_dataloader(
            path_config.data_dir, training_config.batch_size
        )

        data_tensor = torch.load(path_config.data_dir / "imgs.pt")
        model_config.data_shape = data_tensor.shape[1:]

        # Get model
        self.model = AutoEncoder(
            model_config.data_shape,
            model_config.encoder_widths,
            model_config.latent_dim,
        )

        self.model.to(self.device)

        # Define loss, optimizer
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config.lr
        )

        # Further misc steps
        if not os.path.exists(self.path_config.checkpoints_dir):
            os.makedirs(self.path_config.checkpoints_dir)

    def train(self):
        n_batches = len(self.dataloader)
        losses = []

        pbar = tqdm(range(self.n_epochs))

        for epoch in pbar:
            latent_vectors_dict = {}
            epoch_loss = 0

            for img, indices in self.dataloader:
                img = img.to(self.device, non_blocking=True)
                latent_vectors, img_hat = self.model(img)

                reconstruction_loss = self.loss(img, img_hat)

                self.optimizer.zero_grad()
                reconstruction_loss.backward()
                self.optimizer.step()

                latent_vectors_dict.update(
                    {
                        idx: latent.detach().cpu()
                        for idx, latent in zip(indices, latent_vectors)
                    }
                )

                epoch_loss += reconstruction_loss.item()

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if epoch == self.n_epochs - 1 or (
                self.save_ckpt_step is not None
                and epoch != 0
                and epoch % self.save_ckpt_step == 0
            ):
                keys = sorted(latent_vectors_dict.keys())
                latent_tensor = torch.stack(
                    [latent_vectors_dict[k] for k in keys], dim=0
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "latent_vectors": latent_tensor,
                        "avg_losses": losses,
                        "config": asdict(
                            self.model_config,
                        ),
                    },
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                )

            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

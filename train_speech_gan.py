import sys
import logging
import torch
import torchaudio
import librosa
import speechbrain as sb
import matplotlib.pyplot as plt

from pathlib import Path
from pesq import pesq
from speechbrain.nnet.loss import stoi_loss
from speechbrain.utils.data_utils import scalarize
from hyperpyyaml import load_hyperpyyaml
from datasets import prepare_libritts, prepare_vctk


plt.switch_backend("agg")
logger = logging.getLogger(__name__)


# Brain class for neural speech coding training
class NCBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Compute for loss calculation and inference in train stage and valid/test stage.

        Args:
            batch (PaddedBatch): This batch object contains all the relevant tensors for computation.
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns:
            dict: A dictionary with keys {"spec", "wav", ...} with predicted features, using predictions[".."] to refer it.
        """

        batch = batch.to(self.device)
        wavs, lens = batch.wav
        wavs = wavs.unsqueeze(1)

        if stage == sb.Stage.TRAIN:
            # The problem of unequal lengths does not occur when using intercepted segments of a specific length during training.

            # model forward
            output_dict = self.modules.generator(wavs)

        else:
            # Padding
            samples = wavs.shape[-1]
            wavs = torch.nn.functional.pad(
                wavs,
                (0, self.hparams.hop_length - (samples % self.hparams.hop_length)),
                "constant",
            )

            # model forward
            output_dict = self.modules.generator(wavs)

            # Trimming
            output_dict["raw_wav"] = output_dict["raw_wav"][:, :, :samples]
            output_dict["est_wav"] = output_dict["est_wav"][:, :, :samples]

        # HiFiGAN: MSD + MPD
        scores_fake, feats_fake = self.modules.discriminator(
            output_dict["est_wav"].detach()
        )
        scores_real, feats_real = self.modules.discriminator(output_dict["raw_wav"])

        output_dict["scores_fake"], output_dict["feats_fake"] = scores_fake, feats_fake
        output_dict["scores_real"], output_dict["feats_real"] = scores_real, feats_real

        return output_dict

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes and combines generator and discriminator losses
        """
        raw_wav, lens = batch.wav
        raw_wav = raw_wav.unsqueeze(1)

        # GAN
        loss_g = self.hparams.generator_loss(
            predictions["est_wav"],
            predictions["raw_wav"],
            predictions["scores_fake"],
            predictions["feats_fake"],
            predictions["feats_real"],
            predictions["mags_comp"],
            predictions["est_mags_comp"],
            predictions["specs_comp"],
            predictions["est_specs_comp"],
            predictions["feature"],
            predictions["quantized_feature"],
        )
        loss_d = self.hparams.discriminator_loss(
            predictions["scores_fake"], predictions["scores_real"]
        )
        loss = {**loss_g, **loss_d}
        self.last_loss_stats[stage] = scalarize(loss)

        # Log info
        if (
            stage == sb.Stage.VALID
            and self.hparams.epoch_counter.current % self.hparams.valid_epochs == 0
        ) or stage == sb.Stage.TEST:
            self.stoi_metric.append(
                batch.id,
                predictions["est_wav"].squeeze(dim=1),
                raw_wav.squeeze(dim=1),
                lens,
                reduction="batch",
            )
            self.pesq_metric.append(
                batch.id,
                predict=predictions["est_wav"].squeeze(dim=1),
                target=raw_wav.squeeze(dim=1),
                lengths=lens,
            )

        # Log and save in test stage.
        if stage == sb.Stage.TEST and self.step <= self.hparams.log_save:
            self.hparams.tensorboard_train_logger.log_audio(
                f"{batch.id[0]}_raw", raw_wav.squeeze(dim=0), self.hparams.sample_rate
            )
            self.hparams.tensorboard_train_logger.log_audio(
                f"{batch.id[0]}_coded",
                predictions["est_wav"].squeeze(dim=0),
                self.hparams.sample_rate,
            )

            demo_pesq = pesq(
                fs=16000,
                ref=raw_wav[0][0].cpu().numpy(),
                deg=predictions["est_wav"][0][0].cpu().numpy(),
                mode="wb",
            )

            coded_mag, _ = librosa.magphase(
                librosa.stft(
                    predictions["est_wav"].squeeze().cpu().detach().numpy(),
                    n_fft=self.hparams.n_fft,
                    hop_length=self.hparams.hop_length,
                    win_length=self.hparams.win_length,
                )
            )
            raw_mag, _ = librosa.magphase(
                librosa.stft(
                    raw_wav.squeeze().cpu().detach().numpy(),
                    n_fft=self.hparams.n_fft,
                    hop_length=self.hparams.hop_length,
                    win_length=self.hparams.win_length,
                )
            )

            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            librosa.display.specshow(
                librosa.amplitude_to_db(coded_mag),
                cmap="magma",
                y_axis="linear",
                ax=axes[0],
                sr=self.hparams.sample_rate,
            )
            axes[0].set_title(f"coded spec, {demo_pesq}")
            librosa.display.specshow(
                librosa.amplitude_to_db(raw_mag),
                cmap="magma",
                y_axis="linear",
                ax=axes[1],
                sr=self.hparams.sample_rate,
            )
            axes[1].set_title("raw spec")
            plt.tight_layout()
            self.hparams.tensorboard_train_logger.writer.add_figure(
                f"{batch.id[0]}_Spectrogram", fig
            )

            if not Path(self.hparams.samples_folder).exists():
                Path.mkdir(Path(self.hparams.samples_folder))
            torchaudio.save(
                Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_raw.wav"),
                raw_wav.squeeze(dim=0).cpu(),
                self.hparams.sample_rate,
                bits_per_sample=16,
            )
            torchaudio.save(
                Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_coded.wav"),
                predictions["est_wav"].squeeze(dim=0).cpu(),
                self.hparams.sample_rate,
                bits_per_sample=16,
            )

        return loss

    def fit_batch(self, batch):
        """
        Train discriminator and generator adversarially
        """
        batch = batch.to(self.device)
        output_dict = self.compute_forward(batch, sb.Stage.TRAIN)

        # First train the discriminator
        loss_d = self.compute_objectives(output_dict, batch, sb.Stage.TRAIN)["D_loss"]
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # Update adv value for generator training
        scores_fake, feats_fake = self.modules.discriminator(output_dict["est_wav"])
        scores_real, feats_real = self.modules.discriminator(output_dict["raw_wav"])

        output_dict["scores_fake"], output_dict["feats_fake"] = scores_fake, feats_fake
        output_dict["scores_real"], output_dict["feats_real"] = scores_real, feats_real

        # Then train the generator
        loss_g = self.compute_objectives(output_dict, batch, sb.Stage.TRAIN)["G_loss"]
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch"""
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        self.last_loss_stats = {}

        def torch_parameter_transfer(obj, path, device):
            """Non-strict Torch Module state_dict load.

            Loads a set of parameters from path to obj. If obj has layers for which
            parameters can't be found, only a warning is logged. Same thing
            if the path has parameters for layers which don't find a counterpart
            in obj.

            Arguments
            ---------
            obj : torch.nn.Module
                Instance for which to load the parameters.
            path : str
                Path where to load from.

            Returns
            -------
            None
                The object is modified in place.
            """
            state_dict = torch.load(path, map_location=device)
            if self.distributed_launch:
                state_dict = {
                    "module." + k if k[:7] != "module." else k: v
                    for k, v in state_dict.items()
                }
            else:
                state_dict = {
                    k[7:] if k[:7] == "module." else k: v for k, v in state_dict.items()
                }
            incompatible_keys = obj.load_state_dict(state_dict, strict=False)
            for missing_key in incompatible_keys.missing_keys:
                logger.warning(
                    f"During parameter transfer to {obj} loading from "
                    + f"{path}, the transferred parameters did not have "
                    + f"parameters for the key: {missing_key}"
                )
            for unexpected_key in incompatible_keys.unexpected_keys:
                logger.warning(
                    f"During parameter transfer to {obj} loading from "
                    + f"{path}, the object could not use the parameters loaded "
                    + f"with the key: {unexpected_key}"
                )

        if self.hparams.epoch_counter.current == 0 and self.hparams.fine_tune:
            torch_parameter_transfer(
                self.modules.generator,
                self.hparams.trained_generator,
                device=self.device,
            )
            logger.info("Load exited generator.")

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(device=torch.device(self.device))

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            opt_g_class, opt_d_class = self.opt_class
            # opt_g_class, opt_d_class, sch_g_class, sch_d_class = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(self.modules.discriminator.parameters())
            # self.scheduler_g = sch_g_class(self.optimizer_g)
            # self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer_g", self.optimizer_g)
                self.checkpointer.add_recoverable("optimizer_d", self.optimizer_d)
                # self.checkpointer.add_recoverable("scheduler_g", self.scheduler_d)
                # self.checkpointer.add_recoverable("scheduler_d", self.scheduler_d)

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch.

        Args:
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage. Defaults to None.
        """

        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.numpy(),
                deg=pred_wav.numpy(),
                mode="wb",
            )

        if (
            stage == sb.Stage.VALID and epoch % self.hparams.valid_epochs == 0
        ) or stage == sb.Stage.TEST:
            self.stoi_metric = sb.utils.metric_stats.MetricStats(
                metric=stoi_loss.stoi_loss
            )
            self.pesq_metric = sb.utils.metric_stats.MetricStats(
                metric=pesq_eval, n_jobs=self.hparams.pesq_n_jobs, batch_eval=False
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Gets called at the end of an epoch.

        Args:
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
            stage_loss (float): The average loss for all of the data processed in this stage.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage. Defaults to None.
        """

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # Update learning rate
            # self.scheduler_g.step()
            # self.scheduler_d.step()
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]

            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            self.hparams.tensorboard_train_logger.log_stats(
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=self.last_loss_stats[sb.Stage.VALID],
            )
            if epoch % self.hparams.valid_epochs == 0:
                self.valid_stats = {
                    "stoi": -self.stoi_metric.summarize("average"),
                    "pesq": self.pesq_metric.summarize("average"),
                }

                self.valid_stats_tb = {
                    "stoi": [-i for i in self.stoi_metric.scores],
                    "pesq": self.pesq_metric.scores,
                }

                # The train_logger writes a summary to stdout and to the logfile.
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch},
                    valid_stats=self.valid_stats,
                )

                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch},
                    valid_stats=self.valid_stats_tb,
                )

                self.checkpointer.save_and_keep_only(
                    meta=self.valid_stats, num_to_keep=3
                )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            test_stats = {
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=test_stats,
            )

            # self.hparams.tensorboard_train_logger.log_stats(
            #     {"Epoch loaded": self.hparams.epoch_counter.current},
            #     test_stats=test_stats,
            # )


def dataio_prep(hparams):
    """
    This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    Args:
        hparams (dict): This dictionary is loaded from the `train.yaml` file, and it includes all the hyperparameters needed for dataset construction and loading.

    Returns:
        dict: Contains two keys, "train" and "valid" that correspond to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("path", "segment")
    @sb.utils.data_pipeline.provides("wav")
    def audio_pipeline(path, segment):
        wav = sb.dataio.dataio.read_audio(path)
        if segment:
            segment_size = int(hparams["segment_size"] * hparams["sample_rate"])
            if wav.size(0) > segment_size:
                max_audio_start = wav.size(0) - segment_size
                audio_start = torch.randint(0, max_audio_start, (1,))
                wav = wav[audio_start : audio_start + segment_size]
            else:
                wav = torch.nn.functional.pad(
                    wav, (0, segment_size - wav.size(0)), "constant"
                )

        return wav

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "wav"],
            # "id" is implicitly added as an item in the data point.
        )
    return datasets


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        if hparams["dataset"] == "LibriTTS":
            sb.utils.distributed.run_on_main(
                prepare_libritts,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_json_train": hparams["train_annotation"],
                    "save_json_valid": hparams["valid_annotation"],
                    "save_json_test": hparams["test_annotation"],
                    "sample_rate": hparams["sample_rate"],
                    "train_subsets": hparams["train_subsets"],
                    "valid_subsets": hparams["valid_subsets"],
                    "test_subsets": hparams["test_subsets"],
                    "min_duration": hparams["min_duration"],
                },
            )
        elif hparams["dataset"] == "VCTK":
            sb.utils.distributed.run_on_main(
                prepare_vctk,
                kwargs={
                    "data_folder": hparams["data_folder"],
                    "save_json_train": hparams["train_annotation"],
                    "save_json_valid": hparams["valid_annotation"],
                    "save_json_test": hparams["test_annotation"],
                    "sample_rate": hparams["sample_rate"],
                    "mic_id": hparams["mic_id"],
                    "min_duration": hparams["min_duration"],
                },
            )

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    nc_brain = NCBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    nc_brain.fit(
        epoch_counter=nc_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Load best checkpoint (highest SISNR) for evaluation
    test_stats = nc_brain.evaluate(
        test_set=datasets["test"],
        max_key=None,
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

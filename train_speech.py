import sys
import torch
import torchaudio
import librosa
import speechbrain as sb
import matplotlib.pyplot as plt

from pathlib import Path
from pesq import pesq
from speechbrain.nnet.loss import stoi_loss
from hyperpyyaml import load_hyperpyyaml
from datasets import prepare_libritts


plt.switch_backend('agg')

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
            output_dict = self.modules.model(wavs)
            
        else:
            # Padding
            samples = wavs.shape[-1]
            wavs = torch.nn.functional.pad(
                wavs, (0, self.hparams.hop_length - (samples % self.hparams.hop_length)), "constant"
            )
        
            # model forward
            output_dict = self.modules.model(wavs)
            
            # Trimming
            output_dict["raw_wav"] = output_dict["raw_wav"][:, :, :samples]
            output_dict["est_wav"] = output_dict["est_wav"][:, :, :samples]

        return output_dict

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given the predicted and targeted outputs.

        Args:
            predictions (dict): The output dict from `compute_forward`.
            batch (PaddedBatch): This batch object contains all the relevant tensors for computation.
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns:
            torch.Tensor: A one-element tensor used for backpropagating the gradient.
        """

        # Prepare clean targets for comparison
        raw_wav, lens = batch.wav
        raw_wav = raw_wav.unsqueeze(1)

        # Total loss consists of loss_rencon and loss_vq
        loss_recon = self.hparams.compute_cost_recon(predict_dict=predictions)
        loss_commit = self.hparams.compute_cost_commit(predict_dict=predictions)
        loss = loss_recon + loss_commit

        # Log info
        # Train stage: recon loss, commit loss
        # Valid stage: recon loss, commit loss, stoi and pesq
        # Test stage: stoi and pesq
        if stage != sb.Stage.TEST:
            self.loss_recon_metric.append(batch.id, predict_dict=predictions, lens=lens, reduction="batch")
            self.loss_commit_metric.append(batch.id, predict_dict=predictions, lens=lens, reduction="batch")

        if stage != sb.Stage.TRAIN:
            self.stoi_metric.append(batch.id, predictions["est_wav"].squeeze(dim=1), raw_wav.squeeze(dim=1), lens, reduction="batch")

            if stage == sb.Stage.TEST:
                self.pesq_metric.append(batch.id, predict=predictions["est_wav"].squeeze(dim=1), target=raw_wav.squeeze(dim=1), lengths=lens)

        # Log and save in test stage.
        if stage == sb.Stage.TEST and self.step <= self.hparams.log_save:
            
            self.hparams.tensorboard_train_logger.log_audio(f"{batch.id[0]}_raw", raw_wav.squeeze(dim=0), self.hparams.sample_rate)
            self.hparams.tensorboard_train_logger.log_audio(f"{batch.id[0]}_coded", predictions["est_wav"].squeeze(dim=0), self.hparams.sample_rate)
            
            demo_pesq = pesq(fs=16000, ref=raw_wav[0][0].cpu().numpy(), deg=predictions["est_wav"][0][0].cpu().numpy(), mode="wb",)

            coded_mag, _ = librosa.magphase(librosa.stft(predictions["est_wav"].squeeze().cpu().detach().numpy(), n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length, win_length=self.hparams.win_length))
            raw_mag, _ = librosa.magphase(librosa.stft(raw_wav.squeeze().cpu().detach().numpy(), n_fft=self.hparams.n_fft, hop_length=self.hparams.hop_length, win_length=self.hparams.win_length))

            fig, axes = plt.subplots(2, 1, figsize=(6, 6))
            librosa.display.specshow(librosa.amplitude_to_db(coded_mag), cmap="magma", y_axis="linear", ax=axes[0], sr=self.hparams.sample_rate)
            axes[0].set_title(f'coded spec, {demo_pesq}')
            librosa.display.specshow(librosa.amplitude_to_db(raw_mag), cmap="magma", y_axis="linear", ax=axes[1], sr=self.hparams.sample_rate)
            axes[1].set_title('raw spec')
            plt.tight_layout()
            self.hparams.tensorboard_train_logger.writer.add_figure(f'{batch.id[0]}_Spectrogram', fig)

            if not Path(self.hparams.samples_folder).exists():
                Path.mkdir(Path(self.hparams.samples_folder))
            torchaudio.save(Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_raw.wav"), raw_wav.squeeze(dim=0).cpu(), self.hparams.sample_rate, bits_per_sample=16)
            torchaudio.save(Path(self.hparams.samples_folder).joinpath(f"{batch.id[0]}_coded.wav"), predictions["est_wav"].squeeze(dim=0).cpu(), self.hparams.sample_rate, bits_per_sample=16)


        return loss

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch.

        Args:
            stage (sb.Stage): One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
            epoch (int, optional): The currently-starting epoch. This is passed `None` during the test stage. Defaults to None.
        """
        

        # Log the reconstruct and the commit loss in train/valid stage.
        if stage != sb.Stage.TEST:

            self.loss_recon_metric = sb.utils.metric_stats.MetricStats(
                metric=self.hparams.compute_cost_recon
            )
            self.loss_commit_metric = sb.utils.metric_stats.MetricStats(
                metric=self.hparams.compute_cost_commit
            )

        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.numpy(),
                deg=pred_wav.numpy(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            
            self.stoi_metric = sb.utils.metric_stats.MetricStats(metric=stoi_loss.stoi_loss)
            
            if stage == sb.Stage.TEST:
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

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:

            # Define the train's stats as attributes to be counted at the valid stage.
            self.train_stats = {
                "loss_recon": self.loss_recon_metric.summarize("average"),
                "loss_commit": self.loss_commit_metric.summarize("average")
            }
            self.train_stats_tb = {
                "loss_recon": self.loss_recon_metric.scores,
                "loss_commit": self.loss_commit_metric.scores
            }
        # Summarize the statistics from the stage for record-keeping.
            

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
                
            if self.hparams.sched:
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                sb.nnet.schedulers.update_learning_rate(self.optimizer, next_lr)

            valid_stats = {
                "loss_recon": self.loss_recon_metric.summarize("average"),
                "loss_commit": self.loss_commit_metric.summarize("average"), 
                "stoi": -self.stoi_metric.summarize("average")
            }

            valid_stats_tb = {
                "loss_recon": self.loss_recon_metric.scores,
                "loss_commit": self.loss_commit_metric.scores, 
                "stoi": [-i for i in self.stoi_metric.scores]
            }
            
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            self.hparams.tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, 
                train_stats=self.train_stats_tb,
                valid_stats=valid_stats_tb,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best pesq score.
            self.checkpointer.save_and_keep_only(meta=valid_stats, max_keys=["stoi"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:

            test_stats = {
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average")
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
        sb.utils.distributed.run_on_main(
            prepare_libritts,
            kwargs={
                'data_folder':hparams["data_folder"],
                'save_json_train':hparams["train_annotation"],
                'save_json_valid':hparams["valid_annotation"],
                'save_json_test':hparams["test_annotation"],
                "sample_rate": hparams["sample_rate"],
                "train_subsets": hparams["train_subsets"],
                "valid_subsets": hparams["valid_subsets"],
                "test_subsets": hparams["test_subsets"],
                "min_duration": hparams["min_duration"]
            },
        )
        

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    nc_brain = NCBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
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
        max_key="stoi",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
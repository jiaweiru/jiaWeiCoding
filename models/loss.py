import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchaudio import transforms
from speechbrain.nnet.losses import mse_loss, l1_loss


def magri_loss(predict_dict, lens=None, reduction="mean"):
    """
    Compute MagRI loss between compressed predict and target spec.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[2]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) \
            + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)
    
    return loss_mag + loss_ri

def ri_loss(predict_dict, lens=None, reduction="mean"):
    """
    Compute RI loss between compressed predict and target spec.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_spec, target_spec = predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_spec, target_spec = predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    # Use speechbrain style to accommodate variable length training.

    loss_ri = mse_loss(predict_spec, target_spec, length=lens, reduction=reduction)

    return loss_ri

def commit_loss(predict_dict, cost, lens=None, reduction="mean"):
    """
    Compute commitment loss between the input and the quantized feature.
    Note that the quantized feature has no gradient attribute due to detach().

    Args:
        predict_dict (dict): A dictionary of predictions, including mags, specs, ... ,wavs.
        reduction (str, optional): Reduction in speechbrain.nnet.losses.mse_loss. Defaults to "mean".
    """

    feature, feature_q = predict_dict["feature"], predict_dict["quantized_feature"]
    feature, feature_q = feature.permute(0, 3, 1, 2).contiguous(), feature_q.permute(0, 3, 1, 2).contiguous()
    # [B, C, F, T] -> [B, T, C, F]
    # Use speechbrain style to accommodate variable length training.

    return mse_loss(feature, feature_q, length=lens, reduction=reduction) * cost


def tf_loss(predict_dict, lens=None, cost_f=1, cost_t=10, reduction="mean"):
    """
    Compute MagRI loss between compressed predict and target spec, and L1 loss in time domain.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    raw_wav, predict_wav = predict_dict["raw_wav"], predict_dict["est_wav"]
    raw_wav, predict_wav = raw_wav.squeeze(dim=1), predict_wav.squeeze(dim=1)
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[2]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) \
            + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)
    
    loss_t = l1_loss(predict_wav, raw_wav, length=lens, reduction=reduction)

    return cost_f * (loss_mag + loss_ri) + cost_t * loss_t


def magri_feature_loss(predict_dict, cost_f=1, cost_m=0.1, lens=None, reduction="mean"):
    """
    Compute MagRI loss between compressed predict and target spec.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec \
        = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[2]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) \
            + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)

    loss_match = 0.
    encoder_feature = predict_dict["encoder_feature"]
    decoder_feature = list(reversed(predict_dict["decoder_feature"]))

    for e, d in zip(encoder_feature, decoder_feature):
        e, d = e.permute(0, 3, 1, 2).contiguous(), d.permute(0, 3, 1, 2).contiguous()
        loss_match += mse_loss(e, d, length=lens, reduction=reduction)

    return cost_f * (loss_mag + loss_ri) + cost_m * loss_match


def mel_loss(mel_dict, compression, predict_dict, type, lens=None, reduction="mean"):
    """
    Calculation of mel spectrum loss.

    Args:
        mel_dict (dict): A dictionary of mel transform parameters, including sample_rate, hop_length, win_length, ... ,mel_scale.
        compression (bool): Whether to compress it by log-law.
        predict_dict (dict): A dictionary of predictions, including mags, specs, ... ,wavs.
        reduction (str, optional): Reduction in speechbrain.nnet.losses.mse_loss. Defaults to "mean".
    """
    raw_wav, predict_wav = predict_dict["raw_wav"], predict_dict["est_wav"]
    raw_wav, predict_wav = raw_wav.squeeze(dim=1), predict_wav.squeeze(dim=1)
    
    sample_rate, hop_length, win_length, n_fft, n_mels, f_min, f_max, power, normalized, norm, mel_scale \
        = mel_dict["sample_rate"], mel_dict["hop_length"], mel_dict["win_length"], mel_dict["n_fft"], mel_dict["n_mels"], mel_dict["f_min"], mel_dict["f_max"], mel_dict["power"], mel_dict["normalized"], mel_dict["norm"], mel_dict["mel_scale"]
    audio2mel = transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length, win_length=win_length, n_fft=n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max, power=power, normalized=normalized, norm=norm, mel_scale=mel_scale).to(raw_wav.device)
    
    raw_mel = audio2mel(raw_wav).permute(0, 2, 1).contiguous()
    predict_mel = audio2mel(predict_wav).permute(0, 2, 1).contiguous()
    
    if compression:
        # eps = 1e-5, C = 1, similar to HifiGAN
        raw_mel = torch.log(torch.clamp(raw_mel, min=1e-5) * 1)
        predict_mel = torch.log(torch.clamp(predict_mel, min=1e-5) * 1)
    if type == 'l2':
        return mse_loss(predict_mel, raw_mel, length=lens, reduction=reduction)
    elif type == 'l1':
        return l1_loss(predict_mel, raw_mel, length=lens, reduction=reduction)
    

def multimel_loss(mel_dict, level, predict_dict, loss_tp, lens=None, reduction="mean"):
    """
    Calculation of multi resolution mel spectrum loss.
    
    Args:
        mel_dict (dict): A dictionary of mel transform parameters, including only sample_rate, n_mels, f_min, f_max, power, normalized, norm, mel_scale.
        level(tuple): Resolution Level, for example (5, 11) means that the window length is from the 5th power of 2 to the 11th power of 2.
        predict_dict (dict): A dictionary of predictions, including mags, specs, ... ,wavs.
        reduction (str, optional): Reduction in speechbrain.nnet.losses.mse_loss. Defaults to "mean".
    """
    loss = 0.
    for i in range(level[0], level[1] + 1):
        mel_dict["hop_length"] = 2 ** i // 4
        mel_dict["win_length"] = 2 ** i
        mel_dict["n_fft"] = 2 ** i
        loss += mel_loss(mel_dict, True, predict_dict, loss_tp, lens, reduction)
        
    total_loss = loss / (level[1] + 1 - level[0])
    
    return total_loss


def magri_multimel_loss(mel_dict, level, predict_dict, cost_magri, cost_multimel, mel_tp='l2', lens=None, reduction="mean"):
    """
    MagRI + MultiMel Loss
    """
    loss_magri = magri_loss(predict_dict, lens, reduction)
    loss_multimel = multimel_loss(mel_dict, level, predict_dict, mel_tp, lens, reduction)
    
    return loss_magri * cost_magri + loss_multimel * cost_multimel


def ri_multimel_loss(mel_dict, level, predict_dict, cost_ri, cost_multimel, mel_tp='l2', lens=None, reduction="mean"):
    """
    MagRI + MultiMel Loss
    """
    loss_ri = ri_loss(predict_dict, lens, reduction)
    loss_multimel = multimel_loss(mel_dict, level, predict_dict, mel_tp, lens, reduction)
    
    return loss_ri * cost_ri + loss_multimel * cost_multimel


# For fine-tune GAN
class CommitLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, feats, quantized_feats):
        
        return self.loss_func(feats, quantized_feats)


class GeneratorLoss(nn.Module):

    def __init__(
        self,
        stft_loss=None,
        stft_loss_weight=0,
        mseg_loss=None,
        mseg_loss_weight=0,
        feat_match_loss=None,
        feat_match_loss_weight=0,
        l1_spec_loss=None,
        l1_spec_loss_weight=0,
        commit_loss=None,
        commit_loss_weight=0
    ):
        super().__init__()
        self.stft_loss = stft_loss
        self.stft_loss_weight = stft_loss_weight
        self.mseg_loss = mseg_loss
        self.mseg_loss_weight = mseg_loss_weight
        self.feat_match_loss = feat_match_loss
        self.feat_match_loss_weight = feat_match_loss_weight
        self.l1_spec_loss = l1_spec_loss
        self.l1_spec_loss_weight = l1_spec_loss_weight
        self.commit_loss = commit_loss
        self.commit_loss_weight = commit_loss_weight

    def _apply_G_adv_loss(self, scores_fake, loss_func):

        adv_loss = 0
        if isinstance(scores_fake, list):
            for score_fake in scores_fake:
                fake_loss = loss_func(score_fake)
                adv_loss += fake_loss
            # adv_loss /= len(scores_fake)
        else:
            fake_loss = loss_func(scores_fake)
            adv_loss = fake_loss
        return adv_loss
    
    def forward(
        self,
        y_hat=None,
        y=None,
        scores_fake=None,
        feats_fake=None,
        feats_real=None,
        feats=None,
        feats_q=None
    ):

        gen_loss = 0
        adv_loss = 0
        loss = {}

        # STFT Loss
        if self.stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(
                y_hat[:, :, : y.size(2)].squeeze(1), y.squeeze(1)
            )
            loss["G_stft_loss_mg"] = stft_loss_mg
            loss["G_stft_loss_sc"] = stft_loss_sc
            gen_loss = gen_loss + self.stft_loss_weight * (
                stft_loss_mg + stft_loss_sc
            )

        # L1 Spec loss
        if self.l1_spec_loss:
            l1_spec_loss = self.l1_spec_loss(y_hat, y)
            loss["G_l1_spec_loss"] = l1_spec_loss
            gen_loss = gen_loss + self.l1_spec_loss_weight * l1_spec_loss

        # Commitment loss
        if self.commit_loss:
            loss_commit = self.commit_loss(feats, feats_q)
            loss["G_commit_loss"] = loss_commit
            gen_loss = gen_loss + self.commit_loss_weight * loss_commit
        
        # multiscale MSE adversarial loss
        if self.mseg_loss and scores_fake is not None:
            mse_fake_loss = self._apply_G_adv_loss(scores_fake, self.mseg_loss)
            loss["G_mse_fake_loss"] = mse_fake_loss
            adv_loss = adv_loss + self.mseg_loss_weight * mse_fake_loss

        # Feature Matching Loss
        if self.feat_match_loss and feats_fake is not None:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            loss["G_feat_match_loss"] = feat_match_loss
            adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss
        loss["G_loss"] = gen_loss + adv_loss
        loss["G_gen_loss"] = gen_loss
        loss["G_adv_loss"] = adv_loss

        return loss
import torch 
import torch.nn
import torch.nn.functional as F

from speechbrain.nnet.losses import mse_loss, l1_loss


def magri_loss(predict_dict, lens=None, reduction="mean"):
    """
    Compute MagRI loss between compressed predict and target spec.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_mag, target_mag, predict_spec, target_spec = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[1]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)

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
    predict_mag, target_mag, predict_spec, target_spec = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    raw_wav, predict_wav = predict_dict["raw_wav"], predict_dict["est_wav"]
    raw_wav, predict_wav = raw_wav.squeeze(dim=1), predict_wav.squeeze(dim=1)
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[1]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)
    
    loss_t = l1_loss(predict_wav, raw_wav, length=lens, reduction=reduction)

    return cost_f * (loss_mag + loss_ri) + cost_t * loss_t


def magri_feature_loss(predict_dict, cost_f=1, cost_m=0.1, lens=None, reduction="mean"):
    """
    Compute MagRI loss between compressed predict and target spec.

    Args:
        predict_dict(dict): A dictionary of predictions, including mags, specs, ... ,wavs.
    """
    predict_mag, target_mag, predict_spec, target_spec = predict_dict["est_mags_comp"], predict_dict["mags_comp"], predict_dict["est_specs_comp"], predict_dict["specs_comp"]
    predict_mag, target_mag, predict_spec, target_spec = predict_mag.permute(0, 2, 1).contiguous(), target_mag.permute(0, 2, 1).contiguous(), predict_spec.permute(0, 2, 1).contiguous(), target_spec.permute(0, 2, 1).contiguous()
    # Use speechbrain style to accommodate variable length training.

    # the number of freq bins 
    bins = predict_mag.shape[1]

    loss_mag = mse_loss(predict_mag, target_mag, length=lens, reduction=reduction)
    loss_ri = mse_loss(predict_spec[:, :, :bins], target_spec[:, :, :bins], length=lens, reduction=reduction) + mse_loss(predict_spec[:, :, bins:], target_spec[:, :, bins:], length=lens, reduction=reduction)

    loss_match = 0.
    encoder_feature = predict_dict["encoder_feature"]
    decoder_feature = list(reversed(predict_dict["decoder_feature"]))

    for e, d in zip(encoder_feature, decoder_feature):
        e, d = e.permute(0, 3, 1, 2).contiguous(), d.permute(0, 3, 1, 2).contiguous()
        loss_match += mse_loss(e, d, length=lens, reduction=reduction)

    return cost_f * (loss_mag + loss_ri) + cost_m * loss_match


def commit_loss_magphase(predict_dict, cost, lens=None, reduction="mean"):
    """
    Compute commitment loss for mag phase decoupled model between the input and the quantized feature.
    Note that the quantized feature has no gradient attribute due to detach().

    Args:
        predict_dict (dict): A dictionary of predictions, including mags, specs, ... ,wavs.
        reduction (str, optional): Reduction in speechbrain.nnet.losses.mse_loss. Defaults to "mean".
    """

    mags_feature, mags_feature_q = predict_dict["mags_feature"], predict_dict["mags_quantized_feature"]
    mags_feature, mags_feature_q = mags_feature.permute(0, 3, 1, 2).contiguous(), mags_feature_q.permute(0, 3, 1, 2).contiguous()
    phases_feature, phases_feature_q = predict_dict["phases_feature"], predict_dict["phases_quantized_feature"]
    phases_feature, phases_feature_q = phases_feature.permute(0, 3, 1, 2).contiguous(), phases_feature_q.permute(0, 3, 1, 2).contiguous()
    # [B, C, F, T] -> [B, T, C, F]
    # Use speechbrain style to accommodate variable length training.

    return mse_loss(mags_feature, mags_feature_q, length=lens, reduction=reduction) * cost / 2 + mse_loss(phases_feature, phases_feature_q, length=lens, reduction=reduction) * cost / 2
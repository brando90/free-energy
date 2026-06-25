import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from model.model_utils import get_encoded_images

def call_model_forward(hparams, model, input_tokens, start_pos, bsz):
    #TODO eventually add back kv caching, for now start_pos is not supported in baseline transformer and EBT so start_pos can only be 0
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            embeddings = ebt_outputs[0] # dont return a list just return the final predicted embeddings
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, no_randomness = True)
            embeddings = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1)[:, -1] for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else:
        embeddings = model.forward(input_tokens, start_pos = 0, learning = False)
        energies = None
    return embeddings, energies



def generate_video(model, batch, hparams, decode_frames = False):
    batch_size = batch.shape[0]

    assert hparams.infer_max_gen_len <= hparams.context_length, "max generation length must be <= context length"
    assert hparams.infer_video_condition_frames < hparams.infer_max_gen_len, "condition frames must be < max gen length to ensure are generating >=1 frame"

    frame_condition = batch[:, :hparams.infer_video_condition_frames]
    frame_condition = frame_condition.reshape(-1, *frame_condition.shape[2:]) # becomes B*S (S here is condition), C, W, H
    condition_embeddings = get_encoded_images(frame_condition, hparams.backbone_type, model.image_encoder)
    condition_embeddings = condition_embeddings.reshape(batch_size, hparams.infer_video_condition_frames, model.encoder_dim)

    total_len = hparams.infer_max_gen_len
    generated_embeddings = torch.zeros((batch_size, total_len, model.encoder_dim), device=condition_embeddings.device) # B, S, D
    generated_embeddings[:, :hparams.infer_video_condition_frames] = condition_embeddings
    all_energies = []
    prev_pos = 0

    with torch.no_grad(): # by default no grad, although ebt will enable grad
        for cur_pos in range(hparams.infer_video_condition_frames, total_len):
            input_embeddings = generated_embeddings[:, :cur_pos] # NOTE removed prev_pos since are not using start_pos in model forward for now, TODO eventually add back
            if hparams.model_name == "ebt":
                next_embeddings, energies = call_model_forward(hparams, model, input_embeddings, prev_pos, batch_size)
                all_energies.append(energies) # only want last energies
            else:
                next_embeddings, _ = call_model_forward(hparams, model, input_embeddings, prev_pos, batch_size) # no energies for most models

            generated_embeddings[:, cur_pos] = next_embeddings[:, -1]
            prev_pos = cur_pos

    if decode_frames:
        generated_frames = model.image_encoder.decode(generated_embeddings.reshape(-1, 4, 28, 28)).sample 
        generated_frames = generated_frames.reshape(batch_size, total_len, *generated_frames.shape[1:])
        outputs = {"video": generated_frames}
    else:
        outputs = {"video": None}

    gt_frames = batch[:, hparams.infer_video_condition_frames:total_len] # B, S - conditional_frames, C, W, H
    gt_frames = gt_frames.reshape(-1, *gt_frames.shape[2:])
    gt_embeddings = get_encoded_images(gt_frames, hparams.backbone_type, model.image_encoder)
    gt_embeddings = gt_embeddings.reshape(batch_size, total_len - hparams.infer_video_condition_frames, model.encoder_dim)
    gen_embeddings = generated_embeddings[:, hparams.infer_video_condition_frames:] # only grab generated embeddings after conditioning embeddings
    mse_loss = F.mse_loss(gen_embeddings, gt_embeddings)
    smooth_l1_loss = F.smooth_l1_loss(gen_embeddings, gt_embeddings)
    outputs["mse_loss"] = mse_loss
    outputs["smooth_l1_loss"] = smooth_l1_loss

    # DEBUG CODE as a sanity check: log loss on the frame right after the condition ends, as there has been no exposure bias yet
    temp_mse_loss_first_frame = F.mse_loss(generated_embeddings[:, hparams.infer_video_condition_frames], gt_embeddings[:, 0])
    outputs["temp_mse_loss_first_frame"] = temp_mse_loss_first_frame
    temp_smooth_l1_loss_first_frame = F.smooth_l1_loss(generated_embeddings[:, hparams.infer_video_condition_frames], gt_embeddings[:, 0])
    outputs["temp_smooth_l1_loss_first_frame"] = temp_smooth_l1_loss_first_frame
    
    if hparams.model_name == "ebt":
        energy_tensors = []
        for step_energies in zip(*all_energies):
            step_tensor = torch.stack(step_energies)
            avg_step_energy = torch.mean(step_tensor, dim=0)
            energy_tensors.append(avg_step_energy)
        
        for step_idx, energy in enumerate(energy_tensors):
            outputs[f"mcmc_step_{step_idx}_energy"] = energy
    return outputs
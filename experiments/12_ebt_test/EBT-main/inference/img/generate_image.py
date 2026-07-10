import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from model.model_utils import denormalize

#NOTE this is only for denoising from an image for now

def call_model_forward(hparams, model, noised_images, bsz, t):
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(noised_images, learning = False)
            denoised_images = ebt_outputs[0][-1]
        else:
            ebt_outputs = model.forward(noised_images, learning = False, no_randomness = True)
            denoised_images = ebt_outputs[0][-1] # uses 0, -1 since ebt returns tuple of lists of (logits, energy predictions) for each mcmc step; dont want learning mode since needs grad
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1)[:, -1] for energy_tensor in energies] # will be num_mcmc_step * energy landscapes len list, with bsz elements each
    else: # diffusion
        with torch.no_grad():
            t = hparams.infer_increase_steps if hparams.infer_increase_steps != 0 else t[0]
            model_kwargs = dict(y=None)
            denoised_images = model.diffusion.ddim_sample_loop(
                        model.forward, noised_images.shape, noise=noised_images, model_kwargs=model_kwargs, device=model.device, eta=0.0, timestep_start=t # all have same timestep start
                    )
            energies = None
            if hparams.infer_recurse_diffusion_n_times > 0:
                for i in range(hparams.infer_recurse_diffusion_n_times):
                    denoised_images = model.diffusion.ddim_sample_loop(
                            model.forward, noised_images.shape, noise=denoised_images, model_kwargs=model_kwargs, device=model.device, eta=0.0, timestep_start=t # all have same timestep start
                        )
    return denoised_images, energies


def generate_image(model, batch, hparams):
    real_images = batch['image']
    bsz = real_images.shape[0]
    # print("model.diffusion.num_timesteps", model.diffusion.num_timesteps)
    denoising_t_timestep = int(hparams.denoise_images_noise_level * model.diffusion.num_timesteps) - 1
    # print("denoising_t_timestep", denoising_t_timestep, "for denoising image level", hparams.denoise_images_noise_level)
    t = torch.tensor(denoising_t_timestep, device=model.device).expand(bsz)
    noised_images = model.diffusion.q_sample(real_images, t)
    denoised_images, energies = call_model_forward(hparams, model, noised_images, bsz, t)

    denorm_real_images = denormalize(real_images, hparams.dataset_name, model.device, hparams.custom_image_normalization, True).squeeze(0)
    denorm_denoised_images = denormalize(denoised_images, hparams.dataset_name, model.device, hparams.custom_image_normalization, True).squeeze(0)
    
    normalize_img_mse = F.mse_loss(torch.clamp(denorm_denoised_images, 0, 1), denorm_real_images)

    denorm_real_images = torch.clamp(denorm_real_images * 255, 0, 255)
    denorm_denoised_images = torch.clamp(denorm_denoised_images * 255, 0, 255) # confirmed and clamp does help psnr and mse (slightly, +0.01 psnr and -0.3 mse)

    pixel_mse = F.mse_loss(denorm_denoised_images, denorm_real_images)
    max_pixel_value = 255.0
    psnr = 10 * torch.log10((max_pixel_value ** 2) / pixel_mse)

    outputs = {
        "normalize_img_mse" : normalize_img_mse,
        "pixel_mse" : pixel_mse,
        "psnr" : psnr
    }

    if hparams.model_name == "ebt":
        all_energies = [energies] # since generate all patches at once
        energy_tensors = []
        for step_energies in zip(*all_energies):
            step_tensor = torch.stack(step_energies)
            avg_step_energy = torch.mean(step_tensor, dim=0)
            energy_tensors.append(avg_step_energy)
        
        for step_idx, energy in enumerate(energy_tensors):
            outputs[f"mcmc_step_{step_idx}_energy"] = energy

    if model.trainer.global_step % hparams.log_image_every_n_steps == 0:
        denorm_noised_images = denormalize(noised_images, hparams.dataset_name, model.device, hparams.custom_image_normalization, True).squeeze(0)
        outputs['noised_image'] = denorm_noised_images[0]
        outputs['original_image'] = denorm_real_images[0] / 255
        outputs['denoised_image'] = denorm_denoised_images[0] / 255

    
    return outputs
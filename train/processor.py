import random
from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from collections import Counter
import math
@dataclass
class StudentSampleInfo:
    sigmas: torch.Tensor
    window_start_indices: List[int]
    window_end_indices: List[int]

@dataclass
class SampleSigmasInfo:
    sample_step_indices: List[int]
    sample_sigmas: torch.Tensor
    sample_next_sigmas: torch.Tensor
    random_sigmas: torch.Tensor

class DistillProcessor:
    def __init__(
        self,
        shift: float,
        device: torch.device,
        dtype: torch.dtype,
        sampling_steps_list: List[int],
        teacher_training_steps: int = 1000,
        student_shift: Optional[float] = None,
        student_sigmas: Optional[list] = None,
        uncond_model_kwargs: Optional[dict] = None,
        train_window: Optional[list] = None,
        split_interval: Optional[list] = None,
        use_sigma_function: bool = False,
        points_mode: Optional[str] = None,
        cfg_control: float = 1.0,
    ):
        self.device = device
        self.dtype = dtype
        self.teacher_training_steps = teacher_training_steps
        self.uncond_model_kwargs = uncond_model_kwargs
        self.train_window = train_window
        self.cfg_control = cfg_control
        if isinstance(sampling_steps_list, int):
            sampling_steps_list = [sampling_steps_list]
        sampling_steps_list = sorted(list(set(sampling_steps_list)))
        self.sampling_steps_list: List[int] = sampling_steps_list
        # Time axis / sigma axis for teacher
        self.teacher_shift = shift
        sigmas = torch.linspace(1.0, 0, teacher_training_steps + 1, device=device)
        self.teacher_sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        if student_shift is None:
            student_shift = shift
        self.student_shift = student_shift
        self.student_sample_info_dict: Dict[int, StudentSampleInfo] = {}
        self.build_student_sample_info(student_shift, student_sigmas)
        # Set default info
        self.sampling_steps = sampling_steps_list[0]
        self.student_sample_info = self.student_sample_info_dict[self.sampling_steps]
        self.step_choice_counter = Counter()   # cumulative counts for each step index
        self.step_choice_total = 0             # total sampled indices so far
        self.step_choice_print_every = 250     # print every N calls (adjust as you like)
        self.step_choice_call = 0
        # ----- Train windows without fake model -------
        self.train_fake = False
        # ----- sample sigmas from the given prob function ----- 
        self.use_sigma_function = use_sigma_function
        self.random_sigma_function = None
        if self.use_sigma_function:
            # self.random_sigma_function, _ = self.make_random_sigma_function(mode="nonlinear", w0=0.35, sharpness=5.0)
            self.random_sigma_function, _ = self.make_soft_smooth_points_function(points_mode=points_mode)
        elif split_interval is not None:
            self.random_sigma_function = self.make_binary_interval_function(start=split_interval[0], end=split_interval[1])
        else:
            print("Didnt use any soft_smooth_points_function or binary_interval_function!!")
        self.min_step_idx = 0
        self.max_step_idx = 1000

    def make_random_sigma_function(self,
        x_target=650, w_target=0.05, w_min=1e-6,
        split_x=500, w_split=0.25,   
        mode="linear",
        peak_x=100,
        sharpness=1.0,
        w0=0.3
    ):

        k = math.log(w_split / w_target) / (x_target - split_x)

        p = float(peak_x) / float(split_x)
        p = min(max(p, 1e-3), 1 - 1e-3)
        a = p * sharpness + 1.0
        b = (1 - p) * sharpness + 1.0
        t_star = a / (a + b)
        shape_max = (t_star ** a) * ((1 - t_star) ** b) 

        def random_sigma_function(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            w_exp = w_split * torch.exp(-k * (x - float(split_x)))
            if mode == "linear":
                w_0_split = 0.5 + (w_split - 0.5) * (x / float(split_x))
            elif mode == "nonlinear":
                t = (x / float(split_x)).clamp(0, 1)
                shape = (t ** a) * ((1 - t) ** b) / (shape_max + 1e-12)  
                baseline = w0 + (w_split - w0) * t                       
                baseline_at_peak = w0 + (w_split - w0) * t_star
                lift = (0.5 - baseline_at_peak)                      
                w_0_split = baseline + lift * shape
            else:
                raise ValueError(f"Unknown sigma_function mode: {mode}")
            w = torch.where(x <= float(split_x), w_0_split, w_exp)
            return torch.clamp(w, min=w_min)

        return random_sigma_function, k
    
    def make_binary_interval_function(self, start, end, inside=1.0, outside=0.0, inclusive=True):
        start, end = float(start), float(end)
        if end < start:
            raise ValueError(f"end must be >= start, got start={start}, end={end}")

        def f(x):
            x = torch.as_tensor(x, dtype=torch.float32)
            m = (x >= start) & (x <= end) if inclusive else (x > start) & (x < end)
            return torch.where(m, torch.tensor(inside, dtype=torch.float32), torch.tensor(outside, dtype=torch.float32))

        return f

    def make_soft_smooth_points_function(
        self,
        points=None,                # custom points (list of (x,y)) OR None to use points_mode preset
        points_mode="a",            # "a" "b" "c" "d" "e" (used when points is None)
        sigma=100.0,
        w_min=1e-6,
        smooth_in_log=True,
        pad_mode="edge",
    ):
        """
        Very smooth curve from anchor points
        - padding mode controls boundary slope behavior:
            * reflect      -> slope near boundary tends to 0
            * edge         -> constant extension
            * linear_ramp  -> linear extrapolation-like extension
        """

        presets = {
            "a": [(0, 1.0), (250, 0.8), (400, 0.2), (1000, 0.001)],
            "b": [(0, 1.0), (100, 1.0), (250, 1.0), (500, 0.3), (750, 0.05), (1000, 0.001)],
            "c": [(0, 0.6), (300, 1.3), (500, 0.05), (1000, 0.001)],
            "d": [(0, 0.01), (300, 0.1), (500, 0.95), (1000, 1.0)],
            "e": [(0, 0.4), (300, 0.05), (500, 0.95), (1000, 1.0)],
        }

        if points is None:
            points = presets.get(points_mode, [(0, 1.0), (250, 0.9), (500, 0.05), (1000, 0.001)])

        pts = np.array(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("points must be (N,2) with N>=2")

        pts = pts[np.argsort(pts[:, 0])]
        xs, ys = pts[:, 0], pts[:, 1]
        x0, x1 = int(np.floor(xs[0])), int(np.ceil(xs[-1]))
        grid_x = np.arange(x0, x1 + 1, dtype=np.float64)
        grid_y = np.interp(grid_x, xs, ys, left=ys[0], right=ys[-1])

        def _smooth(arr):
            if sigma <= 0:
                return arr
            r = int(max(1, round(3 * sigma)))
            t = np.arange(-r, r + 1, dtype=np.float64)
            k = np.exp(-0.5 * (t / sigma) ** 2)
            k /= (k.sum() + 1e-12)
            arr_pad = np.pad(arr, (r, r), mode=pad_mode)
            return np.convolve(arr_pad, k, mode="valid")

        if smooth_in_log:
            grid_y = np.maximum(grid_y, w_min)
            grid_y = np.exp(_smooth(np.log(grid_y)))
        else:
            grid_y = _smooth(grid_y)

        def f(x):
            x_t = torch.as_tensor(x, dtype=torch.float32)
            x_np = np.clip(x_t.detach().cpu().numpy().astype(np.float64), grid_x[0], grid_x[-1])
            y_np = np.interp(x_np, grid_x, grid_y)
            return torch.clamp(torch.as_tensor(y_np, dtype=torch.float32, device=x_t.device), min=w_min)

        return f, points

    def set_sampling_steps(self, sampling_steps: int):
        self.sampling_steps = sampling_steps
        self.student_sample_info = self.student_sample_info_dict[sampling_steps]
    
    def build_student_sample_info(self, student_shift: float = None, student_sigmas: Optional[list] = None):
        if student_shift is None:
            student_shift = self.teacher_shift
        self.student_shift = student_shift
        for sampling_steps in self.sampling_steps_list:
            if student_sigmas is None:
                sigmas = torch.linspace(1.0, 0, sampling_steps + 1, device=self.device)
                sigmas = student_shift * sigmas / (1 + (student_shift - 1) * sigmas)
            else:
                sigmas = torch.tensor(student_sigmas, device=self.device, dtype=self.dtype)
            print(f"sigmas:{sigmas}")
            indices = (self.teacher_sigmas.unsqueeze(0) - sigmas.unsqueeze(1)).abs().argmin(dim=1)
            student_sigmas = self.teacher_sigmas[indices]
            window_start_indices: List[int] = indices[:sampling_steps]
            window_end_indices: List[int] = indices[-sampling_steps:]
            
            self.student_sample_info_dict[sampling_steps] = StudentSampleInfo(
                student_sigmas,
                window_start_indices,
                window_end_indices,
            )


    def sample_student_sampling_steps(self):
        if len(self.sampling_steps_list) == 1:
            return
        random_idx = torch.randint(
            low=0,
            high=len(self.sampling_steps_list),
            size=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        if dist.is_initialized():
            # Ensure that all processes sample the same number of sampling steps
            # because FSDP model must perform inference over the same number of steps
            dist.barrier()
            dist.broadcast(random_idx, src=0)
        sampling_steps = self.sampling_steps_list[random_idx.item()]
        self.set_sampling_steps(sampling_steps)

    def _step(
        self,
        latents: torch.Tensor,
        velocity: torch.Tensor,
        current_sigmas: torch.Tensor,
        next_sigmas: torch.Tensor,
        eta: float = 1.0,   # eta = 0.0: -> x0 -> add ; eta = 1.0: delta*v
    ) -> torch.Tensor:
        """
        v -> pred_noise -> add -> noise
        current -> clean -> next
        Here we assume that both latents and velocity are in SD3 latent format [B, C, H, W].
        """
        dtype = latents.dtype
        latents = latents.to(torch.float32)
        velocity = velocity.to(torch.float32)
        # [B, 1, 1, 1] for broadcasting with [B, C, H, W]
        current_sigmas_4d = current_sigmas.reshape(-1, 1, 1, 1)
        next_sigmas_4d = next_sigmas.reshape(-1, 1, 1, 1)
        # latents = latents + (next_sigmas_4d - current_sigmas_4d) * velocity
        # clean / pred_noise remain unchanged; only broadcast dimensions changed to 4D
        clean_latents = latents + (0.0 - current_sigmas_4d) * velocity
        pred_noise = latents + (1.0 - current_sigmas_4d) * velocity
        add_noise = eta * pred_noise + ((1 - eta**2) ** 0.5) * torch.randn_like(pred_noise)
        latents = self.add_noise(clean_latents, torch.zeros_like(current_sigmas_4d), next_sigmas_4d, add_noise)
        latents = latents.to(dtype)
        return latents

    def model_sample(
        self,
        model,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        next_sigmas: torch.Tensor,
        model_kwargs: dict,
        eta: float = 1.0,
        cfg: Optional[float] = None,
    ):
        """
        model: SD3 medium transformer / fake_model / real_model
        latents: [B, C, H, W]
        sigmas: [B, 1, 1, 1]  -> reshape(-1)*teacher_training_steps serves as timestep
        """
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            pred = model(
                hidden_states=latents,
                timestep=sigmas.reshape([-1]) * self.teacher_training_steps,
                **model_kwargs,
            )[0]

        if cfg is not None:
            # enable CFG when sigmas < cfg_control 
            mask = sigmas <= self.cfg_control   # shape: [B,1,1,1]  bool
            if mask.any():
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    uncond_pred = model(
                        hidden_states=latents,
                        timestep=sigmas.reshape([-1]) * self.teacher_training_steps,
                        **self.uncond_model_kwargs,
                    )[0]
                pred = torch.where(mask, uncond_pred + cfg * (pred - uncond_pred), pred)

        latents = self._step(latents, pred, sigmas, next_sigmas, eta)
        return pred, latents

    @torch.no_grad()
    def student_generate(
        self,
        model,
        noises: torch.Tensor,
        model_kwargs: dict,
        eta: float = 1.0,
    ):
        """
        Student inference path, used to generate noisy_latent_list / velocity_list.
        Both noises and latents here are in 4D latent format [B, C, H, W].
        """
        latents = noises
        noisy_latent_list = []
        velocity_list = []
        student_sigmas = self.student_sample_info.sigmas  # [T+1]
        for step in range(self.sampling_steps):
            noisy_latent_list.append(latents)
            pred_v, latents = self.model_sample(
                model=model,
                latents=latents,
                sigmas=student_sigmas[step],
                next_sigmas=student_sigmas[step + 1],
                model_kwargs=model_kwargs,
                eta=eta,
            )
            velocity_list.append(pred_v)
        noisy_latent_list.append(latents)
        return noisy_latent_list, velocity_list

    def add_noise(
        self,
        latents: torch.Tensor,
        sigmas1: torch.Tensor,
        sigmas2: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        All sigma values here are assumed to be in shape [B, 1, 1, 1].
        """
        if noise is None:
            noise = torch.randn_like(latents)
        alphas = (1 - sigmas2) / (1 - sigmas1)
        beta = (sigmas2**2 - (sigmas1 * alphas) ** 2) ** 0.5
        latents = latents * alphas + beta * noise
        return latents

    def sample_sigmas(self, bsz: int, sample_in_window: bool = True) -> SampleSigmasInfo:
        student_sigmas = self.student_sample_info.sigmas  # [T+1]
        window_start_indices = self.student_sample_info.window_start_indices
        window_end_indices = self.student_sample_info.window_end_indices

        # -------------------------- Support probability-weighted train_window selection ----------------------------
        if self.train_window is None or self.train_fake:
            sample_step_indices = [random.randint(0, self.sampling_steps - 1) for _ in range(bsz)]
        elif isinstance(self.train_window, dict):
            # e.g. {0:0.5, 1:0.3, 2:0.15, 3:0.05}
            steps = [int(k) for k in self.train_window.keys()]
            weights = [float(self.train_window[str(k)]) for k in steps]
            sample_step_indices = random.choices(steps, weights=weights, k=bsz)
            
        elif isinstance(self.train_window, list):
            sample_step_indices = [random.randint(self.train_window[0], self.train_window[-1]) for _ in range(bsz)]
        elif isinstance(self.train_window, float):
            sample_step_indices = [
                (self.sampling_steps - 1) if random.random() < self.train_window
                else random.randint(0, self.sampling_steps - 2)
                for _ in range(bsz)
            ]
        else:
            raise TypeError(f"Unsupported train_window type: {type(self.train_window)}")
        # sample_step_indices = random.choices(steps, weights=weights, k=bsz)

        # ---- accumulate counts ----
        self.step_choice_counter.update(sample_step_indices)
        self.step_choice_total += len(sample_step_indices)
        self.step_choice_call += 1

        # ---- print (rank0 only, and not every time) ----
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and (self.step_choice_call % self.step_choice_print_every == 0):
            # print rank0 local cumulative stats
            print(f"[train_window stats] total={self.step_choice_total}, counts={dict(sorted(self.step_choice_counter.items()))}")

        # ----------------------------------------------------------------------------------------------------------

        sample_step_indices_tensor = torch.tensor(
            sample_step_indices, dtype=torch.int64, device=self.device
        )
        # [B, 1, 1, 1] for broadcasting with latents
        sample_sigmas = student_sigmas[sample_step_indices_tensor].reshape([-1, 1, 1, 1])
        sample_next_sigmas = student_sigmas[sample_step_indices_tensor + 1].reshape([-1, 1, 1, 1])
        if self.train_fake or (self.random_sigma_function is None):
            random_sigma_indices = [
                random.randint(
                    max(window_start_indices[idx] if sample_in_window else 0, self.min_step_idx),
                    min(window_end_indices[idx] - 1, self.max_step_idx)
                )
                for idx in sample_step_indices
            ]
        else:
            random_sigma_indices = []
            for idx in sample_step_indices:
                start = max(window_start_indices[idx] if sample_in_window else 0, self.min_step_idx)
                end = min(window_end_indices[idx] - 1, self.max_step_idx)
                candidates = list(range(start, end + 1))
                weights = self.random_sigma_function(candidates)
                if isinstance(weights, torch.Tensor):
                    weights = weights.detach().float().cpu().tolist()
                else:
                    weights = [float(w) for w in weights]
                random_sigma_indices.append(random.choices(candidates, weights=weights, k=1)[0])
                # print(f"weights:{weights}; start:{start}; end:{end}; random_sigma_indices:{random_sigma_indices}")
                # breakpoint()

        random_sigma_indices_tensor = torch.as_tensor(random_sigma_indices, dtype=torch.int64, device=self.device)
        random_sigmas = self.teacher_sigmas[random_sigma_indices_tensor].reshape([-1, 1, 1, 1])
        return SampleSigmasInfo(sample_step_indices, sample_sigmas, sample_next_sigmas, random_sigmas)

    @torch.no_grad()
    def sample_for_train_fake_model(
        self,
        bsz: int,
        noisy_latent_list: List[torch.Tensor],
        velocity_list: List[torch.Tensor],
        sample_in_window: bool = False,
    ):
        """
        Returns:
            Noised latent samples, corresponding sigmas, and student predictions.
        """
        self.train_fake = True
        sample_info = self.sample_sigmas(bsz=bsz, sample_in_window=sample_in_window)
        self.train_fake = False
        # Window start point latents
        noisy_latents_ode_startpoint = torch.stack(
            [noisy_latent_list[sample_info.sample_step_indices[i]][i] for i in range(bsz)],
            dim=0,
        )
        velocity = torch.stack(
            [velocity_list[sample_info.sample_step_indices[i]][i] for i in range(bsz)],
            dim=0,
        )
        # Window end point latents
        noisy_latents_ode = torch.stack(
            [noisy_latent_list[sample_info.sample_step_indices[i] + 1][i] for i in range(bsz)],
            dim=0,
        )
        noisy_model_latents = self.add_noise(
            noisy_latents_ode,
            sample_info.sample_next_sigmas,
            sample_info.random_sigmas,
        )
        # Explicitly pass zero sigma tensor for compatibility with 4D assumption in `_step` function
        zero_sigmas = torch.zeros_like(sample_info.sample_sigmas)
        clean_sample = self._step(
            noisy_latents_ode_startpoint,
            velocity,
            sample_info.sample_sigmas,
            zero_sigmas,
        )
        target = (noisy_model_latents - clean_sample) / sample_info.random_sigmas
        return noisy_model_latents, sample_info.random_sigmas, target

    def sample_for_train_student_model(
        self,
        model,
        bsz: int,
        noisy_latent_list: List[torch.Tensor],
        model_kwargs: dict,
        sample_in_window: bool = False,
        eta: float = 1.0,
    ):
        """
        Returns:
            Student-predicted endpoint latents starting from window start point,
            noised latents,
            sigma of noisy latents,
            sigma of window endpoint latents
        """
        sample_info = self.sample_sigmas(bsz=bsz, sample_in_window=sample_in_window)
        noisy_latents_ode = torch.stack(
            [noisy_latent_list[sample_info.sample_step_indices[i]][i] for i in range(bsz)],
            dim=0,
        )
        # Support gradient computation for student model
        _, noisy_model_latents_ode = self.model_sample(
            model=model,
            latents=noisy_latents_ode,
            sigmas=sample_info.sample_sigmas,
            next_sigmas=sample_info.sample_next_sigmas,
            model_kwargs=model_kwargs,
            eta=eta,
        )
        noisy_model_latents = self.add_noise(
            noisy_model_latents_ode,
            sample_info.sample_next_sigmas,
            sample_info.random_sigmas,
        )
        return (
            noisy_model_latents_ode,
            noisy_model_latents,
            sample_info.random_sigmas,
            sample_info.sample_next_sigmas,
        )

    def train_discriminator(
        self,
        bsz: int,
        clean_img_latents: torch.Tensor,
        noisy_latent_list: List[torch.Tensor],
        model_kwargs: dict,
        model: torch.nn.Module,
        block_ids: list,
        loss_weights: float = 1.0,
        gradient_accumulation_steps: int = 1,
        sample_in_window: bool = True,
    ) -> torch.Tensor:
        sample_info = self.sample_sigmas(bsz=bsz, sample_in_window=sample_in_window)
        # Endpoint latents within the window
        noisy_latents_ode = torch.stack(
            [noisy_latent_list[sample_info.sample_step_indices[i] + 1][i] for i in range(bsz)],
            dim=0,
        )
        noisy_latents = self.add_noise(
            noisy_latents_ode,
            sigmas1=sample_info.sample_next_sigmas,
            sigmas2=sample_info.random_sigmas,
        )
        noise = torch.randn_like(clean_img_latents)
        # [B, 1, 1, 1] for broadcasting to [B, C, H, W]  
        random_sigmas_4d = sample_info.random_sigmas.to(self.dtype)
        noisy_clean_latents = noise * random_sigmas_4d + (1 - random_sigmas_4d) * clean_img_latents
        # Loss computed separately to save memory
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            real_logits = model(
                block_ids=block_ids,
                hidden_states=noisy_clean_latents.detach(),
                timestep=sample_info.random_sigmas.reshape([-1])*self.teacher_training_steps,
                **model_kwargs,
            )
        r_loss = torch.mean(F.relu(1.0 - real_logits)) * loss_weights
        r_loss = r_loss / gradient_accumulation_steps
        r_loss.backward()
        r_loss = r_loss.detach()
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            fake_logits = model(
                block_ids=block_ids,
                hidden_states=noisy_latents.detach(),
                timestep=sample_info.random_sigmas.reshape([-1])*self.teacher_training_steps,
                **model_kwargs,
            )
        f_loss = torch.mean(F.relu(1.0 + fake_logits)) * loss_weights
        f_loss = f_loss / gradient_accumulation_steps
        f_loss.backward()
        f_loss = f_loss.detach()
        d_loss = r_loss + f_loss
        print(f"real:{torch.mean(F.relu(1.0 - real_logits))}")
        print(f"fake:{torch.mean(F.relu(1.0 + fake_logits))} ")
        return d_loss

    def train_generator(
        self,
        generator,
        discriminator,
        bsz: int,
        noisy_latent_list: List[torch.Tensor],
        model_kwargs: dict,
        block_ids: list,
        loss_weights: float = 1.0,
        gradient_accumulation_steps: int = 1,
        sample_in_window: bool = True,
    ) -> torch.Tensor:
        sample_info = self.sample_sigmas(bsz=bsz, sample_in_window=sample_in_window)
        noisy_latents_ode = torch.stack(
            [noisy_latent_list[sample_info.sample_step_indices[i]][i] for i in range(bsz)],
            dim=0,
        )
        # Support gradient information for generator
        _, noisy_latents_ode = self.model_sample(
            model=generator,
            latents=noisy_latents_ode,
            sigmas=sample_info.sample_sigmas,
            next_sigmas=sample_info.sample_next_sigmas,
            model_kwargs=model_kwargs,
        )
        noisy_latents = self.add_noise(
            noisy_latents_ode,
            sigmas1=sample_info.sample_next_sigmas,
            sigmas2=sample_info.random_sigmas,
        )
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            fake_logits = discriminator(
                block_ids=block_ids,
                hidden_states=noisy_latents,
                timestep=sample_info.random_sigmas.reshape([-1])*self.teacher_training_steps,
                **model_kwargs,
            )
        g_loss = -torch.mean(fake_logits) * loss_weights
        g_loss = g_loss / gradient_accumulation_steps
        g_loss.backward()
        return g_loss.detach()

if __name__ == "__main__":
    from pprint import pprint
    solver = DistillProcessor(
        shift=3.0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        sampling_steps_list=[4],
        student_shift=3.0,
    )
    pprint(solver.student_sample_info_dict)
    teacher_sigmas = solver.teacher_sigmas
    window_start_indices = solver.student_sample_info.window_start_indices
    window_end_indices = solver.student_sample_info.window_end_indices
    for i in range(solver.sampling_steps_list[0]):
        start_sigma = teacher_sigmas[window_start_indices[i]]
        end_sigma = teacher_sigmas[window_end_indices[i]]
        print(f"window[{i}] start sigma:{start_sigma}, end sigma:{end_sigma}")
    info = solver.sample_sigmas(2)
    print(info.random_sigmas)
    print(torch.clamp(info.random_sigmas - 0.1, min=0))
    print(info.random_sigmas.reshape([-1]))
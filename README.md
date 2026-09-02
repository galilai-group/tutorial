# Tiny LeJEPA and LeWM tutorials

- `inet10.py` trains a ResNet9 LeJEPA on 64×64 ImageNette.
- `mmnist.py` trains an action-conditioned latent world model on Moving MNIST.
- `app.py` provides an interactive open-loop Moving-MNIST viewer.

## Install

```bash
pip install -e .
```

CUDA runs use BF16 and compile the expensive encoder path automatically.

## ImageNette LeJEPA

```bash
python inet10.py
```

The default is the measured speed/accuracy configuration: ResNet9, 29 epochs,
64×64 images, three augmented views, 1,024 SIGReg directions, constant-LR
AdamW at `1.8e-3`, and a stable-pretraining online linear probe at `3e-3`.

On one datacenter GPU it reached **76.7% final / 78.1% peak probe accuracy in
8m29s**. It uses 16 persistent data workers by default; set `--workers` to the
number of CPUs assigned to the job.

The explicit full command for the roughly 80%-in-8-minutes run is:

```bash
python inet10.py --epochs 29 --views 3 --slices 1024 \
  --lr 1.8e-3 --probe-lr 3e-3 --lamb 0.01 --batch-size 256 --workers 16
```

```bash
python inet10.py --steps 10 --workers 0  # pipeline smoke test
```

## Moving-MNIST LeWM

```bash
python mmnist.py
```

Each sample contains one digit moving under observed `(delta_x, delta_y)`
actions. Training uses a ResNet9 encoder, recursive open-loop prediction loss,
and SIGReg over pooled trajectory states. Three detached stable-pretraining
callbacks monitor the representation throughout training:

- linear and two-layer MLP digit probes (`eval/digit_probe_acc` and
  `eval/digit_mlp_probe_acc`);
- linear and two-layer MLP position probes (`eval/position_probe_r2` and
  `eval/position_mlp_probe_r2`);
- an online image decoder (`eval/decoder_MeanSquaredError`).

The final `RESULT` line reports both probe scores and rollout MSE. Each run also
saves `lewm.pt`, one teacher-forcing-free GIF per class (`rollout_0.gif` through
`rollout_9.gif`), and a ten-row comparison in `rollout.gif`.

The default is the clearest translation-only sweep configuration: 100 epochs,
batch size 128, eight-step trajectories, 64 latent dimensions, pooled `(N*T,D)`
SIGReg with λ `0.1` and 2,048 directions, AdamW at `1e-3`, probe LR `3e-3`, and
rollout weight `1.0`. It completed in **12m28s** on one datacenter GPU. Its final
linear/nonlinear digit accuracies were **66.4% / 76.9%**, position R² scores were
**0.9195 / 0.9830**, latent rollout MSE was **0.1247**, and decoder MSE was
**0.00531**. The explicit equivalent command is:

```bash
python mmnist.py --epochs 100 --samples 10000 --batch-size 128 --workers 8 \
  --rollout 8 --rotation-step 0 --latent-dim 64 --lr 1e-3 --probe-lr 3e-3 \
  --lamb 0.1 --rollout-weight 1 --sigreg-mode pooled --slices 2048
```

Enable the experimental third action dimension with discrete
`-45° / 0° / +45°` rotations:

```bash
python mmnist.py --rotation-step 45
```

```bash
python mmnist.py --steps 10 --workers 0 --samples 512 --no-compile  # smoke test
```

## Interactive world-model app

After at least one Moving-MNIST run, launch:

```bash
python app.py
```

The app automatically loads the curated checkpoint in `default_checkpoint.txt`
(falling back to the newest `lewm.pt` below `runs/`) and listens on
`0.0.0.0:8000`. Open `http://<compute-node>:8000`. From another machine, forward
the port and open `http://localhost:8000`:

```bash
ssh -L 8000:localhost:8000 <compute-node>
```

Pass a checkpoint only when you want a specific run:

```bash
python app.py path/to/lewm.pt
```

Only reset encodes a true image. Every arrow-key or button action after reset
recursively advances the previous imagined latent. The interface displays the
truth, decoded imagination, action history, and complete open-loop trajectory.

## Outputs

Both training scripts use stable-pretraining's run manager and write
`metrics.csv`, frozen requirements, environment metadata, and artifacts below:

```text
runs/runs/<date>/<time>/<run-id>/
```

Use `--output PATH` to choose another root. Each script prints its exact output
paths in a final `RESULT` line.

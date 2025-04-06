# MusicGen Trainer

This is a trainer for MusicGen model. It's based on [this](https://github.com/chavinlo/musicgen_trainer).

# Contributors
- [@mkualquiera](https://github.com/mkualquiera) and [@neverix](https://github.com/neverix): actually got it working
- elyxlz: help with masks

## STATUS: MVP

Removing the gradient scaler, increasing the batch size and only training on conditional samples makes training work.

TODO:
* [ ] Add notebook
* [ ] Add webdataset support
* [ ] Try larger models
* [ ] Add LoRA
* [ ] Make rolling generation customizable

## Usage

### Dataset Creation

Create a folder, in it, place your audio and caption files. **They must be `.wav` and `.txt` format respectively.** You can omit `.txt` files for training with empty text by setting the `--no_label` option to `1`.

![](https://i.imgur.com/AlDlqBI.png)

You can use `.wav` files longer than 30 seconds, in that case the model will be trained on random crops of the original `.wav` file.

In this example, segment_000.txt contains the caption "jazz music, jobim" for wav file segment_000.wav.

### Running the trainer

Run `python3 run.py --dataset <PATH_TO_YOUR_DATASET>`. Make sure to use the full path to the dataset, not a relative path.

### Options

- `dataset_path`: String, path to your dataset with `.wav` and `.txt` pairs.
- `model_id`: String, MusicGen model to use. Can be `small`/`medium`/`large`. Default: `small`
- `lr`: Float, learning rate. Default: `0.00001`/`1e-5`
- `epochs`: Integer, epoch count. Default: `100`
- `use_wandb`: Integer, `1` to enable wandb, `0` to disable it. Default: `0` = Disabled
- `save_step`: Integer, amount of steps to save a checkpoint. Default: None
- `no_label`: Integer, whether to read a dataset without `.txt` files. Default: `0` = Disabled
- `tune_text`: Integer, perform textual inversion instead of full training. Default: `0` = Disabled
- `weight_decay`: Float, the weight decay regularization coefficient. Default: `0.00001`/`1e-5`
- `grad_acc`: Integer, number of steps to smooth gradients over. Default: 2
- `warmup_steps`: Integer, amount of steps to slowly increase learning rate over to let the optimizer compute statistics. Default: 16
- `batch_size`: Integer, batch size the model sees at once. Reduce to lower memory consumption. Default: 4
- `use_cfg`: Integer, whether to train with some labels randomly dropped out. Default: `0` = Disabled

You can set these options like this: `python3 run.py --use_wandb=1`.

### Models

Once training finishes, the model (and checkpoints) will be available under the `models` folder in the same path you ran the trainer on.

![](https://i.imgur.com/Mu19EPb.png)

To load them, simply run the following on your generation script:

```python
model.lm.load_state_dict(torch.load('models/lm_final.pt'))
```

Where `model` is the MusicGen Object and `models/lm_final.pt` is the path to your model (or checkpoint).

## Citations

```
@article{copet2023simple,
      title={Simple and Controllable Music Generation},
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2023},
      journal={arXiv preprint arXiv:2306.05284},
}
```

@mkualquiera (mkualquiera@discord) added batching, debugged the code and trained the first working model.

Special thanks to elyxlz (223864514326560768@discord) for helping @chavinlau with the masks.

@chavinlau wrote the original version of the training code. Original README:

---

# MusicGen Trainer

This is a trainer for MusicGen model. Currently it's very basic but I'll add more features soon.

## STATUS: BROKEN

Only works for overfitting. Breaks model on anything else

More information on the current training quality on the [experiments section](#experiments)

## Usage

### Dataset Creation

Create a folder, in it, place your audio and caption files. **They must be WAV and TXT format respectively.**

![](https://i.imgur.com/AlDlqBI.png)

### Important: Split your audios in 35 second chunks. Only the first 30 seconds will be processed. Audio cannot be less than 30 seconds.

In this example, segment_000.txt contains the caption "jazz music, jobim" for wav file segment_000.wav

### Running the trainer

Run `python3 run.py --dataset /home/ubuntu/dataset`, replace `/home/ubuntu/dataset` with the path to your dataset. Make sure to use the full path, not a relative path.

### Options

- `dataset_path`: String, path to your dataset with WAV and TXT pairs.
- `model_id`: String, MusicGen model to use. Can be `small`/`medium`/`large`. Default: `small`
- `lr`: Float, learning rate. Default: `0.0001`/`1e-4`
- `epochs`: Integer, epoch count. Default: `5`
- `use_wandb`: Integer, `1` to enable wandb, `0` to disable it. Default: `0` = Disabled
- `save_step`: Integer, amount of steps to save a checkpoint. Default: None

You can set these options like this: `python3 run.py --use_wandb=1`

### Models

Once training finishes, the model (and checkpoints) will be available under the `models` folder in the same path you ran the trainer on.

![](https://i.imgur.com/Mu19EPb.png)

To load them, simply run the following on your generation script:

```python
model.lm.load_state_dict(torch.load('models/lm_final.pt'))
```

Where `model` is the MusicGen Object and `models/lm_final.pt` is the path to your model (or checkpoint).

## Experiments

### Electronic music (Moe Shop):

Encodec seems to struggle with electronic music. Even just Encoding->Decoding has many problems.

4:00 - 4:30 - [Moe Shop - WONDER POP](https://youtu.be/H4PZ7mju5QQ?t=240)

Original: https://voca.ro/1jbsor6BAyLY

Encode -> Decode: https://voca.ro/1kF2yyGyRn0y

Overfit -> Generate -> Decode: https://voca.ro/1f6ru5ieejJY

### Bossa Nova (Tom Jobim):

Softer and less aggressive melodies seem to play best with encodec and musicgen. One of these are bossa nova, which to me sounds great:

1:20 - 1:50 - [Tom Jobim - Children's Games](https://youtu.be/8KVtgzOTqDw?t=80)

Original: https://voca.ro/1dm9QpRqa5rj (last 5 seconds are ignored)

Encode -> Decode: https://voca.ro/19LpwVE44si7

Overfit -> Generate -> Decode: https://voca.ro/1hJGVdxsvBOG

## Citations

```
@article{copet2023simple,
      title={Simple and Controllable Music Generation},
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2023},
      journal={arXiv preprint arXiv:2306.05284},
}
```

Special thanks to elyxlz (223864514326560768@discord) for helping me with the masks.

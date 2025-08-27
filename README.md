# Universal audio synthesizer control with normalizing flows

This repository hosts code and additional results for the paper [Universal audio synthesizer control with normalizing flows](https://arxiv.org/abs/1907.00971). You can check out the video demonstration of the FlowSynth on Youtube

[![](https://img.youtube.com/vi/UufQwUitBIw/0.jpg)](https://www.youtube.com/watch?v=UufQwUitBIw)

[https://www.youtube.com/watch?v=UufQwUitBIw](https://www.youtube.com/watch?v=UufQwUitBIw)

## Installing the flow synthesizer plugin

In order to try out the _Flow synthesizer_ plugin, you must 
1. Have an installed version of the [Diva VST](https://u-he.com/products/diva/) (The system works with the free tryout version but will produce noise every now and then). For simplicity, please ensure that it is located here
```
/Library/Audio/Plug-Ins/VST/u-he/Diva.vst
```
2. Install the latest (bleeding-edge) version of both the [Bach and Dada library](https://www.bachproject.net/dl/) for MaxMsp
3. Install the [Mubu library](https://forum.ircam.fr/projects/detail/mubu/) for MaxMsp
4. Have an updated version of `Python 3.7` or newer (tested up to `Python 3.12`)
5. Install the Python dependencies by running the following line at the root of this folder
```bash
$ pip install -r requirements.txt
```
6. Put the `plugin/flow_synth.amxd` device inside a MIDI track in `Ableton Live`
7. ???
7. Profit

NB: If the device seems non-responding, you can try to run the server manually
```bash
$ cd code && python osc_launch.py
```

Note that the plugin has been tested on macOS High Sierra (10.13.6) and is fully compatible with Apple Silicon Macs (M1/M2/M3 series).

## Supporting webpage

For a better viewing experience, please **visit the corresponding [supporting website](https://acids-ircam.github.io/flow_synthesizer/ "Flow synthesizer")**.

It embeds the following:
  * Supplementary figures
  * Audio examples
	* Reconstruction
	* Macro-control learning
	* Neighborhood exploration
	* Interpolation
	* Vocal sketching
  * Real-time implementation in Ableton Live
  
You can also directly parse through the different sub-directories of the main [`docs`](docs) directory.

## Dataset

The dataset can be downloaded here: [https://nubo.ircam.fr/index.php/s/nL3NQomqxced6eJ](https://nubo.ircam.fr/index.php/s/nL3NQomqxced6eJ)

## Code

### Dependencies

#### Python

Code has been developed with `Python 3.7` and is compatible with newer versions up to `Python 3.12`. It has been tested on both Intel and Apple Silicon Macs. Moreover, we rely on several third-party libraries, listed in [`requirements.txt`](requirements.txt). They can be installed with

```bash
$ pip install -r requirements.txt
```

As our experiments are coded in PyTorch, no additional library is required to run them on GPU (provided you already have CUDA installed). On Apple Silicon Macs, PyTorch will automatically utilize optimized Metal Performance Shaders (MPS) when available for improved performance.


#### Audio Rendering

For people interested in the research aspects of this repository, if you want to try new models or evaluate variations of the existing ones, you will need at one point to render the corresponding audio. We now use [DawDreamer](https://github.com/DBraun/DawDreamer) for batch audio generation from synthesizer presets, which provides modern VST3 support and cross-platform compatibility.

To render a dataset with VST3 synthesizers like Massive X:

```bash
python code/render_dataset_dd.py \
  "/Library/Audio/Plug-Ins/VST3/Massive X.vst3" \
  "/path/to/presets" \
  "/path/to/output"
```

See [`docs/dawdreamer_migration.md`](docs/dawdreamer_migration.md) for detailed information about the migration from RenderMan to DawDreamer.

### Train on any AU/VST3 synth (e.g., ZENOLOGY)

You can build a dataset from any installed plugin and train the model on it:

1) Build the dataset with DawDreamer

```
python code/build_plugin_dataset.py \
  --plugin "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component" \
  --out datasets/zenology \
  --count 2000 \
  --keep 64
```

This creates `datasets/zenology/{raw,mel,mfcc}/` with `params.txt` and a `parameter_index_map.json`.

2) Train on the new dataset

```
python code/train.py \
  --path datasets \
  --dataset zenology \
  --data mel \
  --model vae_flow \
  --loss mse \
  --epochs 200 \
  --batch_size 64 \
  --device cpu
```

Notes:
- Prefer VST3 paths when available (e.g., `/Library/Audio/Plug-Ins/VST3/ZENOLOGY.vst3`).
- If the plugin requires licensing, ensure it’s authorized in your OS host.
- Training/evaluation works with the standard mel/MFCC specs used in this repo.

### Usage

The code is mostly divided into two scripts `train.py` and `evaluate.py`. The first script `train.py` allows to train a model from scratch as described in the paper. The second script `evaluate.py` allows to generate the figures of the papers, and also all the supporting additional materials visible on the [supporting page](https://acids-ircam.github.io/flow_synthesizer)) of this repository.

#### train.py arguments
```

```

## Pre-trained models

Note that a set of pre-trained models are availble in the `code/results`  folder.

### Models details

As discussed in the paper, the very large amount of baseline models implemented did not allow to provide all the parameters for reference models (which are defined in the source code). However, we provide these details inside the documentation page in the [models details section](https://acids-ircam.github.io/flow_synthesizer/#models-details)

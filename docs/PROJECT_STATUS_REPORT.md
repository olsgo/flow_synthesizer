# Flow Synthesizer Project Status Report

## Project Overview

The Flow Synthesizer is a machine learning project that aims to synthesize audio parameters for the UAD PolyMAX synthesizer using variational autoencoders (VAE) and normalizing flows. The system learns to map mel-spectrogram features from audio to synthesizer parameters.

## Current Architecture

### Model Components
- **Encoder**: `GatedCNN` - Convolutional neural network with gated activations
- **Decoder**: `DecodeCNN` - Transposed convolutional decoder
- **VAE**: Variational autoencoder combining encoder/decoder with latent space
- **Flow Model**: Normalizing flows for parameter generation (planned)

### Data Pipeline
- **Input**: Mel-spectrograms (128 x 173 dimensions, 1 channel)
- **Output**: 66-dimensional synthesizer parameter vectors
- **Dataset**: Located in `/Users/gjb/Datasets/polymax/render`
- **Preprocessing**: Normalization with computed statistics

## Recent Issues Resolved

### 1. Channel Mismatch Error
**Problem**: Decoder expected 1 channel input but received 16 channels
**Solution**: Replaced custom encoder/decoder creation with proper `construct_encoder_decoder` function from `models.basic`
**Files Modified**: `comprehensive_training_fix.py`

### 2. Boolean Tensor Ambiguity
**Problem**: "Boolean value of Tensor with more than one value is ambiguous" errors
**Root Cause**: Direct comparisons of multi-dimensional tensors with scalars
**Solution**: 
- Added proper tensor-to-scalar conversions using `.item()` and `.mean().item()`
- Fixed adaptive loss scaling logic
- Updated outlier detection comparisons

### 3. Gradient Computation Error
**Problem**: "grad can be implicitly created only for scalar outputs"
**Solution**: Ensured loss tensor is scalar before calling `.backward()` by adding mean reduction

### 4. Running Loss Tensor Mismatch
**Problem**: Exponential moving average failed due to different batch sizes
**Solution**: Converted running loss tracking to use scalar values instead of tensors

## Current Working Files

### Main Training Script
- **File**: `comprehensive_training_fix.py`
- **Status**: ✅ Working and stable
- **Features**:
  - Stabilized training with outlier detection
  - Adaptive loss scaling
  - Exponential moving average for loss smoothing
  - Proper tensor health checking
  - Debug logging for model forward passes

### Model Definitions
- **File**: `code/models/basic.py`
- **Contains**: `GatedCNN`, `DecodeCNN`, `construct_encoder_decoder`
- **Status**: ✅ Stable

### VAE Implementation
- **File**: `code/models/vae/vae.py`
- **Status**: ✅ Working with proper encoder dimensions

## Current Training Results

**Latest Run Results**:
- Train Loss: 0.431090
- Validation Loss: 0.347663
- No skipped batches
- Successful forward passes through GatedCNN
- Stable gradient computation

## Project Structure

```
flow_synthesizer/
├── code/
│   ├── models/
│   │   ├── basic.py          # Core CNN models
│   │   ├── vae/              # VAE implementations
│   │   └── flows/            # Flow models (future)
│   ├── utils/                # Data utilities
│   └── train.py              # Original training script
├── comprehensive_training_fix.py  # Current working training script
├── params/                   # Synthesizer presets (100+ files)
├── datasets/                 # Training data
└── outputs/                  # Model checkpoints and results
```

## Next Steps for Development

### Immediate Priorities
1. **Model Evaluation**: Implement comprehensive evaluation metrics
2. **Hyperparameter Tuning**: Optimize learning rates, loss weights, architecture
3. **Data Augmentation**: Expand training dataset
4. **Checkpoint Saving**: Add model persistence

### Medium-term Goals
1. **Flow Integration**: Implement normalizing flows for parameter generation
2. **Real-time Inference**: Optimize for live audio processing
3. **Parameter Validation**: Ensure generated parameters are within valid ranges
4. **Audio Quality Metrics**: Implement perceptual loss functions

### Long-term Vision
1. **VST Integration**: Connect with UAD PolyMAX VST
2. **Real-time Performance**: Live audio-to-parameter conversion
3. **User Interface**: Web-based or standalone application
4. **Multi-synth Support**: Extend to other synthesizers

## Technical Specifications

### Dependencies
- PyTorch (with MPS support for Apple Silicon)
- torchaudio for audio processing
- NumPy, matplotlib for data handling
- pedalboard for VST integration (future)

### Hardware Requirements
- Apple Silicon Mac (current setup uses MPS device)
- Minimum 8GB RAM for training
- UAD PolyMAX VST3 plugin installed

### Key Configuration
- **Input Size**: [1, 128, 173] (mel-spectrogram dimensions)
- **Latent Dimensions**: 35
- **Output Size**: 66 (synthesizer parameters)
- **Batch Size**: 16 (adjustable based on memory)

## Known Issues and Limitations

1. **Dataset Size**: Currently limited training data
2. **Parameter Mapping**: Need validation of parameter ranges
3. **Audio Quality**: No perceptual loss functions yet
4. **Real-time Performance**: Not optimized for live use

## Development Environment Setup

```bash
# Navigate to project
cd /Users/gjb/Projects/flow_synthesizer

# Install dependencies
pip install -r requirements.txt

# Run current training script
python comprehensive_training_fix.py

# Check background services
# TCP server running on terminal 7: python simple_tcp_server.py
```

## Contact and Handover Notes

- **VST Location**: `/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3`
- **Presets Location**: `/Library/Audio/Presets/UADx PolyMAX Synth`
- **Dataset Location**: `/Users/gjb/Datasets/polymax/render`
- **Reference Implementation**: `/Users/gjb/Projects/poc-audio-pedalboard`

## Recent Commit Summary

The project has been stabilized with working training loops. All major tensor dimension and gradient computation issues have been resolved. The codebase is ready for feature development and model improvements.

---

**Report Generated**: Current project state as of latest development session
**Status**: ✅ Training pipeline functional and stable
**Next Developer**: Ready for feature development and optimization

## Updates This Session

- Implemented best-checkpoint saving to `outputs_optimized/models/best_model.pth` during stabilized training and optimized training runs.
- Hardened loss logging to safely handle tensor scalars in `comprehensive_training_fix.py`.
- Added predictor API wrappers `predict_from_file` and `predict_from_audio` with confidence and timing for compatibility with tests and web API.
- Made Ableton integration tolerant to missing model for connection-only tests; clearer messages when model is not loaded.
- Predictor now aligns mel-spectrogram dimensions to the dataset’s expected input size and prefers `params_schema.json` for parameter names.
- Regression head improvements:
  - Added sigmoid output activation (with final clamp) to keep predictions in [0,1] with usable gradients.
  - Added variability-weighted SmoothL1 regression loss using dataset `final_std` to emphasize informative parameters.
  - Applied the same bounding/weighting in `DisentanglingAE` paths.
  - Added BCE loss term for clearly binary toggles (`arp_enable`, `lfo_sync`, `osc_2_sync`, `mod_fx_enable`, `master_bypass`).
- Evaluation robustness:
  - Clamp predictions before metrics.
  - Guard R²/correlation for constant targets; skip zero-range parameters in relative error; use NaN-safe aggregation.

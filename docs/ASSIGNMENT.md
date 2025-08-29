# Flow Synthesizer Development Assignment

## Project Overview

The Flow Synthesizer is a deep learning system that learns to map audio spectrograms to synthesizer parameters, enabling automatic parameter estimation from audio input. The system uses a combination of convolutional neural networks for audio feature extraction and regression models for parameter prediction.

### Core Architecture

- **Audio Processing**: Mel-spectrogram extraction from audio files
- **Feature Extraction**: GatedCNN encoder for latent representation learning
- **Parameter Regression**: MLP-based regression model for synthesizer parameter prediction
- **Target Synthesizer**: UAD PolyMAX Synth (66 parameters)

## Recent Critical Fix ✅

**IMPORTANT**: A major zero-output bug has been recently resolved. The issue was a mismatch between:

- Dataset normalization (z-score: mean=0, std=1, range ≈[-3,+5])
- Model activations (Sigmoid/Hardtanh constraining outputs to [0,1])

**Files Modified**:

- `code/models/basic.py`: Removed `nn.Sigmoid()` from MLP regressor
- `code/models/vae/ae.py`: Removed `Hardtanh(0,1)` from RegressionAE
- `code/evaluate.py`: Fixed evaluation saving logic

**Result**: Model now produces meaningful parameter predictions across the full normalized range.

## Current Project Status

### ✅ Completed Components

1. **Data Pipeline**: PolyMAX dataset loading and preprocessing
2. **Model Architecture**: CNN encoder + MLP regressor
3. **Training Infrastructure**: Optimized training scripts
4. **Evaluation Framework**: Comprehensive evaluation metrics
5. **Zero-Output Fix**: Activation function corrections

### 🔄 Current Development Priorities

#### 1. **Model Training & Optimization** (HIGH PRIORITY)

- **Objective**: Train the fixed model and achieve good parameter prediction accuracy
- **Tasks**:
  - Run full training with the corrected activation functions
  - Monitor training convergence and parameter prediction quality
  - Tune hyperparameters (learning rate, batch size, architecture)
  - Implement early stopping and model checkpointing

#### 2. **Audio Synthesis Integration** (HIGH PRIORITY)

- **Objective**: Complete the audio synthesis pipeline for end-to-end evaluation
- **Current Issue**: Missing synthesizer engine setup in evaluation scripts
- **Tasks**:
  - Fix `AttributeError: 'Namespace' object has no attribute 'engine'` in evaluation
  - Integrate UAD PolyMAX VST3 synthesis using pedalboard
  - Reference implementation: `/Users/gjb/Projects/poc-audio-pedalboard`
  - Enable audio rendering for predicted parameters

#### 3. **Evaluation & Metrics** (MEDIUM PRIORITY)

- **Objective**: Comprehensive model performance assessment
- **Tasks**:
  - Run parameter accuracy evaluation (R², MAE, MSE per parameter)
  - Execute perceptual quality evaluation (spectral similarity)
  - Perform interpolation quality testing
  - Analyze parameter space coverage and dataset biases

#### 4. **Model Architecture Improvements** (MEDIUM PRIORITY)

- **Objective**: Enhance model capacity and performance
- **Tasks**:
  - Experiment with different encoder architectures
  - Implement attention mechanisms for better feature extraction
  - Add regularization techniques (dropout, batch norm)
  - Explore multi-task learning approaches

## Key Files & Directories

### Core Implementation

- `code/models/basic.py`: Model architecture definitions
- `code/models/vae/ae.py`: Autoencoder and regression models
- `code/utils/data.py`: Dataset loading and preprocessing
- `code/train_optimized.py`: Main training script
- `code/evaluate.py`: Evaluation framework

### Data & Assets

- `datasets/polymax_dataset/`: PolyMAX audio and parameter data
- `params/`: Synthesizer preset files (.json)
- `/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3`: Target VST3 plugin
- `/Library/Audio/Presets/UADx PolyMAX Synth`: Factory presets

### Configuration

- VST3 Location: `/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3`
- Preset Location: `/Library/Audio/Presets/UADx PolyMAX Synth`
- Rendered Audio Target: `/Users/gjb/Datasets/polymax/render`

## Development Environment

### Dependencies

- PyTorch (deep learning framework)
- torchaudio (audio processing)
- pedalboard (VST plugin hosting)
- numpy, scipy (numerical computing)
- librosa (audio analysis)

### Hardware Requirements

- GPU recommended for training (CUDA/MPS support)
- Audio interface for real-time synthesis testing
- Sufficient storage for audio datasets

## Getting Started

1. **Environment Setup**:

   ```bash
   cd /Users/gjb/Projects/flow_synthesizer
   pip install -r requirements.txt
   ```

2. **Verify Fix**:

   - The zero-output issue has been resolved
   - Models now produce non-zero parameter predictions
   - Training should converge properly

3. **Next Steps**:
   - Start with training the corrected model
   - Fix synthesis integration issues
   - Run comprehensive evaluations

## Success Metrics

### Short-term (1-2 weeks)

- [ ] Successful model training with positive R² scores
- [ ] Fixed audio synthesis pipeline
- [ ] Basic parameter prediction accuracy > 70%

### Medium-term (1 month)

- [ ] Perceptual quality evaluation showing audio similarity
- [ ] Real-time parameter estimation demo
- [ ] Comprehensive evaluation report

### Long-term (2-3 months)

- [ ] Production-ready model with high accuracy
- [ ] User interface for audio-to-parameters conversion
- [ ] Documentation and deployment pipeline

## Contact & Resources

- **Project Repository**: `/Users/gjb/Projects/flow_synthesizer`
- **Reference Implementation**: `/Users/gjb/Projects/poc-audio-pedalboard`
- **Documentation**: `docs/` directory
- **Previous Evaluation Results**: `evaluation_results/`

---

**Note**: The recent activation function fix was critical - the model was previously constrained to output only values in [0,1] while the dataset parameters span [-3,+5]. This has been resolved, and the model now produces meaningful predictions. Focus on training and synthesis integration as the next priorities.

# Audio Generation Critical Notes

## IMPORTANT: Do NOT Break These Settings Again!

### Sample Rate Configuration

**CRITICAL**: The sample rate MUST be 44100 Hz throughout the entire pipeline.

#### Key Files and Settings:

1. **polymax_dataset_generator.py** (Line ~445)
   ```python
   parser.add_argument(
       '--sample-rate',
       type=int,
       default=44100,  # MUST BE 44100, NOT 22050!
       help='Sample rate for audio generation'
   )
   ```
   - **NEVER** change this back to 22050 Hz
   - This was the root cause of silent audio generation

2. **poc_polymax_loader.py** (Reference Implementation)
   - Uses 44100 Hz sample rate consistently
   - This is the working reference that should be maintained

### Audio Generation Pipeline

#### Working Configuration:
- **Sample Rate**: 44100 Hz
- **Duration**: 4.0 seconds
- **MIDI Note**: C4 (60)
- **Velocity**: 100
- **Note Duration**: 3.0 seconds

#### Spectral Analysis Settings:
- **Mel Spectrograms**: n_mels=128, hop_length=512, n_fft=2048
- **MFCCs**: n_mfcc=13, hop_length=512, n_fft=2048

### What Was Broken and Fixed:

1. **Issue**: Audio files were generated at 22050 Hz instead of 44100 Hz
2. **Symptom**: Silent or very quiet audio with poor quality
3. **Root Cause**: Default sample rate in argument parser was set to 22050
4. **Fix**: Changed default sample rate to 44100 Hz in polymax_dataset_generator.py
5. **Verification**: All 190 presets now generate proper audio with good RMS values

### Testing Checklist:

Before making any changes to audio generation:

1. **Verify Sample Rate**: Check that all components use 44100 Hz
2. **Test Single Preset**: Run with one preset and verify audio quality
3. **Check RMS Values**: Ensure generated audio has reasonable RMS (> 0.001)
4. **Verify Non-Zero Samples**: Audio should have >95% non-zero samples
5. **Compare with Reference**: Use poc_polymax_loader.py as reference

### Command for Quick Testing:
```bash
python polymax_dataset_generator.py --preset-file "/Library/Audio/Presets/UADx PolyMAX Synth/Bells/Clockwork.vstpreset" --output-dir "/Users/gjb/Datasets/polymax_test" --sample-rate 44100
```

### Red Flags - Stop if You See These:

- Audio RMS values consistently < 0.001
- Audio files with >90% zero samples
- Sample rate mismatches between components
- Silent audio output

### Success Indicators:

- Audio RMS values typically 0.01-0.1 range
- 99%+ non-zero samples in audio
- Proper spectral content in Mel/MFCC files
- Consistent 44100 Hz sample rate throughout

---

**Remember**: The poc_polymax_loader.py file is the working reference implementation. When in doubt, check how it handles sample rates and audio generation!
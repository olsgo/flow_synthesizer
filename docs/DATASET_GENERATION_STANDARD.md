# Standard Dataset Generation Methodology

## Overview
This document establishes the proven methodology for generating high-quality audio datasets from VST synthesizers for flow synthesizer training.

## Key Principle: Individual Processing

**ALWAYS process presets individually, never in batch mode.**

This approach has been validated with UAD PolyMAX and should be applied to all synthesizer dataset generation.

## Why Individual Processing?

### Problems with Batch Processing:
1. **State Interference**: Plugin state from previous presets can contaminate subsequent renders
2. **Leftover Tails**: Audio artifacts from previous presets may bleed into new renders
3. **Parameter Drift**: Accumulated parameter changes can affect preset authenticity
4. **Memory Issues**: Plugin memory may not be properly cleared between presets

### Benefits of Individual Processing:
1. **Clean State**: Each preset starts with a fresh plugin instance
2. **Authentic Audio**: No cross-contamination between presets
3. **Consistent Quality**: Reliable, repeatable results
4. **Better Debugging**: Issues can be isolated to specific presets

## Standard Implementation

### Core Approach
```python
# Process each preset in complete isolation
for preset_file in preset_files:
    # Launch fresh process for each preset
    subprocess.run(['python', 'loader.py', '--preset-file', preset_file])
    # Small delay to ensure clean state
    time.sleep(0.2)
```

### Key Requirements
1. **Fresh Process**: Each preset should be processed in a separate subprocess
2. **Complete Isolation**: No shared plugin instances between presets
3. **State Reset**: Plugin should be fully reinitialized for each preset
4. **Verification**: Audio quality should be analyzed after generation

## Quality Validation

### Audio Analysis Metrics
- **RMS Level**: Ensure adequate signal strength
- **Tail Analysis**: Check for leftover artifacts in audio tail
- **Amplitude Spikes**: Detect potential pops/clicks
- **Signal Consistency**: Verify authentic preset characteristics

### Success Criteria
- No leftover tails from previous presets
- Clean audio without artifacts
- Authentic preset sound characteristics
- Consistent quality across all presets

## Proven Results

### UAD PolyMAX Case Study
- **Total Presets**: 190
- **Success Rate**: 100% with individual processing
- **Quality**: 76.2% excellent/good quality (144/189 files)
- **Issues Resolved**: Eliminated state interference and leftover tails

## Application to Other Synthesizers

This methodology should be applied to:
- Serum
- Diva
- Massive
- Any VST synthesizer dataset generation

## Implementation Notes

1. **Performance**: Individual processing takes longer but ensures quality
2. **Monitoring**: Include progress tracking for large preset collections
3. **Error Handling**: Implement timeout and retry mechanisms
4. **Validation**: Always perform post-generation quality analysis

## Conclusion

Individual preset processing is the **gold standard** for synthesizer dataset generation. While it requires more time, it guarantees clean, authentic audio suitable for flow synthesizer training.

**Never compromise on quality for speed when generating training datasets.**
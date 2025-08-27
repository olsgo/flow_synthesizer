#!/usr/bin/env python3
"""
Serum 2 Parameter Mapping
Maps between preset JSON parameter names and VST parameter names.
"""

# Mapping from preset JSON parameter names to VST parameter names
# Format: "preset_param_name": "vst_param_name"
SERUM2_PARAMETER_MAPPING = {
    # Envelopes (Env0 = Env 1, Env1 = Env 2, etc.)
    ".Env0.kParamAttack": "Env 1 Attack",
    ".Env0.kParamDecay": "Env 1 Decay", 
    ".Env0.kParamSustain": "Env 1 Sustain",
    ".Env0.kParamRelease": "Env 1 Release",
    ".Env0.kParamHold": "Env 1 Hold",
    ".Env0.kParamCurve1": "Env 1 Atk Curve",
    ".Env0.kParamCurve2": "Env 1 Dec Curve",
    ".Env0.kParamCurve3": "Env 1 Rel Curve",
    
    ".Env1.kParamAttack": "Env 2 Attack",
    ".Env1.kParamDecay": "Env 2 Decay",
    ".Env1.kParamSustain": "Env 2 Sustain",
    ".Env1.kParamRelease": "Env 2 Release",
    ".Env1.kParamHold": "Env 2 Hold",
    ".Env1.kParamStart": "Env 2 Start",
    ".Env1.kParamEnd": "Env 2 End",
    ".Env1.kParamCurve1": "Env 2 Atk Curve",
    ".Env1.kParamCurve2": "Env 2 Dec Curve",
    ".Env1.kParamCurve3": "Env 2 Rel Curve",
    
    ".Env2.kParamAttack": "Env 3 Attack",
    ".Env2.kParamDecay": "Env 3 Decay",
    ".Env2.kParamSustain": "Env 3 Sustain",
    ".Env2.kParamRelease": "Env 3 Release",
    ".Env2.kParamHold": "Env 3 Hold",
    ".Env2.kParamStart": "Env 3 Start",
    ".Env2.kParamEnd": "Env 3 End",
    ".Env2.kParamCurve1": "Env 3 Atk Curve",
    ".Env2.kParamCurve2": "Env 3 Dec Curve",
    ".Env2.kParamCurve3": "Env 3 Rel Curve",
    
    ".Env3.kParamAttack": "Env 4 Attack",
    ".Env3.kParamDecay": "Env 4 Decay",
    ".Env3.kParamSustain": "Env 4 Sustain",
    ".Env3.kParamRelease": "Env 4 Release",
    ".Env3.kParamHold": "Env 4 Hold",
    ".Env3.kParamStart": "Env 4 Start",
    ".Env3.kParamEnd": "Env 4 End",
    ".Env3.kParamCurve1": "Env 4 Atk Curve",
    ".Env3.kParamCurve2": "Env 4 Dec Curve",
    ".Env3.kParamCurve3": "Env 4 Rel Curve",
    
    # Legacy envelope names for compatibility
    ".Envelope1.kParamAttack": "Env 1 Attack",
    ".Envelope1.kParamDecay": "Env 1 Decay", 
    ".Envelope1.kParamSustain": "Env 1 Sustain",
    ".Envelope1.kParamRelease": "Env 1 Release",
    ".Envelope1.kParamHold": "Env 1 Hold",
    ".Envelope1.kParamAttackCurve": "Env 1 Atk Curve",
    ".Envelope1.kParamDecayCurve": "Env 1 Dec Curve",
    ".Envelope1.kParamReleaseCurve": "Env 1 Rel Curve",
    
    ".Envelope2.kParamAttack": "Env 2 Attack",
    ".Envelope2.kParamDecay": "Env 2 Decay",
    ".Envelope2.kParamSustain": "Env 2 Sustain",
    ".Envelope2.kParamRelease": "Env 2 Release",
    ".Envelope2.kParamHold": "Env 2 Hold",
    ".Envelope2.kParamStart": "Env 2 Start",
    ".Envelope2.kParamEnd": "Env 2 End",
    ".Envelope2.kParamAttackCurve": "Env 2 Atk Curve",
    ".Envelope2.kParamDecayCurve": "Env 2 Dec Curve",
    ".Envelope2.kParamReleaseCurve": "Env 2 Rel Curve",
    
    ".Envelope3.kParamAttack": "Env 3 Attack",
    ".Envelope3.kParamDecay": "Env 3 Decay",
    ".Envelope3.kParamSustain": "Env 3 Sustain",
    ".Envelope3.kParamRelease": "Env 3 Release",
    ".Envelope3.kParamHold": "Env 3 Hold",
    ".Envelope3.kParamStart": "Env 3 Start",
    ".Envelope3.kParamEnd": "Env 3 End",
    ".Envelope3.kParamAttackCurve": "Env 3 Atk Curve",
    ".Envelope3.kParamDecayCurve": "Env 3 Dec Curve",
    ".Envelope3.kParamReleaseCurve": "Env 3 Rel Curve",
    
    ".Envelope4.kParamAttack": "Env 4 Attack",
    ".Envelope4.kParamDecay": "Env 4 Decay",
    ".Envelope4.kParamSustain": "Env 4 Sustain",
    ".Envelope4.kParamRelease": "Env 4 Release",
    ".Envelope4.kParamHold": "Env 4 Hold",
    ".Envelope4.kParamStart": "Env 4 Start",
    ".Envelope4.kParamEnd": "Env 4 End",
    ".Envelope4.kParamAttackCurve": "Env 4 Atk Curve",
    ".Envelope4.kParamDecayCurve": "Env 4 Dec Curve",
    ".Envelope4.kParamReleaseCurve": "Env 4 Rel Curve",
    
    # Oscillators
    ".Oscillator1.kParamEnable": "Osc A On",
    ".Oscillator1.kParamVolume": "A Level",
    ".Oscillator1.kParamPan": "A Pan",
    ".Oscillator1.kParamTune": "A Coarse",
    ".Oscillator1.kParamFineTune": "A Fine",
    ".Oscillator1.kParamPhase": "A Phase",
    ".Oscillator1.kParamRand": "A Rand",
    ".Oscillator1.kParamRetrig": "A Retrig",
    ".Oscillator1.kParamUnison": "A Unison",
    ".Oscillator1.kParamUnisonDetune": "A Detune",
    ".Oscillator1.kParamUnisonBlend": "A Blend",
    
    ".Oscillator2.kParamEnable": "Osc B On",
    ".Oscillator2.kParamVolume": "B Level",
    ".Oscillator2.kParamPan": "B Pan",
    ".Oscillator2.kParamTune": "B Coarse",
    ".Oscillator2.kParamFineTune": "B Fine",
    ".Oscillator2.kParamPhase": "B Phase",
    ".Oscillator2.kParamRand": "B Rand",
    ".Oscillator2.kParamRetrig": "B Retrig",
    ".Oscillator2.kParamUnison": "B Unison",
    ".Oscillator2.kParamUnisonDetune": "B Detune",
    ".Oscillator2.kParamUnisonBlend": "B Blend",
    
    ".Oscillator3.kParamEnable": "Osc C On",
    ".Oscillator3.kParamVolume": "C Level",
    ".Oscillator3.kParamPan": "C Pan",
    ".Oscillator3.kParamTune": "C Coarse",
    ".Oscillator3.kParamFineTune": "C Fine",
    ".Oscillator3.kParamPhase": "C Phase",
    ".Oscillator3.kParamRand": "C Rand",
    ".Oscillator3.kParamRetrig": "C Retrig",
    ".Oscillator3.kParamUnison": "C Unison",
    ".Oscillator3.kParamUnisonDetune": "C Detune",
    ".Oscillator3.kParamUnisonBlend": "C Blend",
    
    # Filters
    ".Filter1.kParamCutoff": "Filter Cutoff",
    ".Filter1.kParamResonance": "Filter Reso",
    ".Filter1.kParamDrive": "Filter Drive",
    ".Filter1.kParamFat": "Filter Fat",
    ".Filter1.kParamKeytrack": "Filter Key",
    ".Filter1.kParamVeltrack": "Filter Vel",
    ".Filter1.kParamType": "Filter Type",
    ".Filter1.kParamSlope": "Filter Slope",
    
    ".Filter2.kParamCutoff": "Filter 2 Cutoff",
    ".Filter2.kParamResonance": "Filter 2 Reso",
    ".Filter2.kParamDrive": "Filter 2 Drive",
    ".Filter2.kParamFat": "Filter 2 Fat",
    ".Filter2.kParamKeytrack": "Filter 2 Key",
    ".Filter2.kParamVeltrack": "Filter 2 Vel",
    ".Filter2.kParamType": "Filter 2 Type",
    ".Filter2.kParamSlope": "Filter 2 Slope",
    
    # LFOs
    ".LFO1.kParamRate": "LFO 1 Rate",
    ".LFO1.kParamShape": "LFO 1 Shape",
    ".LFO1.kParamPhase": "LFO 1 Phase",
    ".LFO1.kParamMode": "LFO 1 Mode",
    ".LFO1.kParamTrigger": "LFO 1 Trig",
    
    ".LFO2.kParamRate": "LFO 2 Rate",
    ".LFO2.kParamShape": "LFO 2 Shape",
    ".LFO2.kParamPhase": "LFO 2 Phase",
    ".LFO2.kParamMode": "LFO 2 Mode",
    ".LFO2.kParamTrigger": "LFO 2 Trig",
    
    ".LFO3.kParamRate": "LFO 3 Rate",
    ".LFO3.kParamShape": "LFO 3 Shape",
    ".LFO3.kParamPhase": "LFO 3 Phase",
    ".LFO3.kParamMode": "LFO 3 Mode",
    ".LFO3.kParamTrigger": "LFO 3 Trig",
    
    ".LFO4.kParamRate": "LFO 4 Rate",
    ".LFO4.kParamShape": "LFO 4 Shape",
    ".LFO4.kParamPhase": "LFO 4 Phase",
    ".LFO4.kParamMode": "LFO 4 Mode",
    ".LFO4.kParamTrigger": "LFO 4 Trig",
    
    # Master/Global
    ".Master.kParamVolume": "Master",
    ".Master.kParamPan": "Master Pan",
    ".Master.kParamPitchBend": "Pitch Bend",
    ".Master.kParamPortamento": "Portamento",
    ".Master.kParamVoices": "Voices",
    ".Master.kParamBendRange": "Bend Range",
    
    # Effects (FX Bus parameters)
    ".FXBus1.kParamLevel": "FX Bus 1 Level",
    ".FXBus1.kParamParam1": "FX Bus 1 Param 1",
    ".FXBus1.kParamParam2": "FX Bus 1 Param 2",
    ".FXBus1.kParamParam3": "FX Bus 1 Param 3",
    ".FXBus1.kParamParam4": "FX Bus 1 Param 4",
    ".FXBus1.kParamParam5": "FX Bus 1 Param 5",
    ".FXBus1.kParamParam6": "FX Bus 1 Param 6",
    ".FXBus1.kParamParam7": "FX Bus 1 Param 7",
    ".FXBus1.kParamParam8": "FX Bus 1 Param 8",
    
    ".FXBus2.kParamLevel": "FX Bus 2 Level",
    ".FXBus2.kParamParam1": "FX Bus 2 Param 1",
    ".FXBus2.kParamParam2": "FX Bus 2 Param 2",
    ".FXBus2.kParamParam3": "FX Bus 2 Param 3",
    ".FXBus2.kParamParam4": "FX Bus 2 Param 4",
    ".FXBus2.kParamParam5": "FX Bus 2 Param 5",
    ".FXBus2.kParamParam6": "FX Bus 2 Param 6",
    ".FXBus2.kParamParam7": "FX Bus 2 Param 7",
    ".FXBus2.kParamParam8": "FX Bus 2 Param 8",
    
    ".FXMain.kParamParam1": "FX Main Param 1",
    ".FXMain.kParamParam2": "FX Main Param 2",
    ".FXMain.kParamParam3": "FX Main Param 3",
    ".FXMain.kParamParam4": "FX Main Param 4",
    ".FXMain.kParamParam5": "FX Main Param 5",
    ".FXMain.kParamParam6": "FX Main Param 6",
    ".FXMain.kParamParam7": "FX Main Param 7",
    ".FXMain.kParamParam8": "FX Main Param 8",
}

def map_preset_to_vst_parameters(preset_data, vst_param_name_to_index):
    """
    Map preset parameters to VST parameters.
    
    Args:
        preset_data: Dictionary of preset parameters
        vst_param_name_to_index: Dictionary mapping VST parameter names to indices
    
    Returns:
        List of tuples (vst_param_index, value) for successfully mapped parameters
    """
    mapped_params = []
    unmapped_params = []
    
    for preset_param, value in preset_data.items():
        # Skip non-parameter data
        if not preset_param.startswith('.'):
            continue
            
        # Check if we have a mapping for this parameter
        if preset_param in SERUM2_PARAMETER_MAPPING:
            vst_param_name = SERUM2_PARAMETER_MAPPING[preset_param]
            
            # Check if the VST parameter exists
            if vst_param_name in vst_param_name_to_index:
                vst_param_index = vst_param_name_to_index[vst_param_name]
                
                # Validate parameter value (should be 0.0 to 1.0)
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    mapped_params.append((vst_param_index, float(value)))
                else:
                    print(f"⚠️  Invalid value for {preset_param} -> {vst_param_name}: {value}")
            else:
                print(f"⚠️  VST parameter not found: {vst_param_name}")
        else:
            unmapped_params.append(preset_param)
    
    if unmapped_params:
        print(f"ℹ️  Unmapped parameters ({len(unmapped_params)}): {unmapped_params[:5]}{'...' if len(unmapped_params) > 5 else ''}")
    
    return mapped_params

def get_mapping_coverage(preset_data):
    """
    Get statistics on mapping coverage for a preset.
    
    Args:
        preset_data: Dictionary of preset parameters
    
    Returns:
        Dictionary with mapping statistics
    """
    preset_params = [k for k in preset_data.keys() if k.startswith('.')]
    mapped_params = [k for k in preset_params if k in SERUM2_PARAMETER_MAPPING]
    
    return {
        'total_preset_params': len(preset_params),
        'mapped_params': len(mapped_params),
        'coverage_percent': (len(mapped_params) / len(preset_params) * 100) if preset_params else 0,
        'unmapped_params': [k for k in preset_params if k not in SERUM2_PARAMETER_MAPPING]
    }

if __name__ == "__main__":
    # Test the mapping with a sample preset
    import json
    import os
    
    preset_path = "converted_presets/Piano/PN - Piano Classic Layer.json"
    if os.path.exists(preset_path):
        with open(preset_path, 'r') as f:
            preset_json = json.load(f)
        
        # Extract the actual parameter data from dawdreamer_params
        preset_data = preset_json.get('dawdreamer_params', {})
        
        stats = get_mapping_coverage(preset_data)
        print(f"Mapping Coverage for {preset_path}:")
        print(f"  Total preset parameters: {stats['total_preset_params']}")
        print(f"  Mapped parameters: {stats['mapped_params']}")
        print(f"  Coverage: {stats['coverage_percent']:.1f}%")
        print(f"  Unmapped examples: {stats['unmapped_params'][:10]}")
        
        # Show some examples of mapped parameters
        mapped_examples = [k for k in preset_data.keys() if k.startswith('.') and k in SERUM2_PARAMETER_MAPPING]
        print(f"  Mapped examples: {mapped_examples[:5]}")
    else:
        print(f"Preset file not found: {preset_path}")
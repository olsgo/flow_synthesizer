#!/usr/bin/env python3
"""
Serum 2 Parameter Mapping

Purpose
- Provide a robust mapping + normalization from Serum 2 preset JSON (dawdreamer_params)
  to DawDreamer/VST parameter indices and normalized values.

Notes
- DawDreamer expects normalized values in [0.0, 1.0]. Serum JSON contains a mix of
  normalized values, percentages, seconds, enumerations, and raw units (Hz, dB, etc.).
- We combine: (1) explicit manual mappings for high-confidence pairs, (2) rule-based
  inference for common patterns (Env, Oscillator, Filter, Global, Macro), and
  (3) conservative skipping of risky/unknown parameters to avoid glitchy audio.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# Mapping from preset JSON parameter names to VST parameter names
# Format: "preset_param_name": "vst_param_name"
SERUM2_PARAMETER_MAPPING: Dict[str, str] = {
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

BLACKLIST_PREFIXES: Tuple[str, ...] = (
    # Non-synthesis editors or transport helpers we should not touch
    ".ClipPlayer",
    ".MidiClip",
    ".ArpClip",
    ".Arp0",
    ".WTOsc",           # wavetable data, indexes, etc.
    ".GranularOsc",     # internal granular UI state
    ".FXRack",          # FX contain many non-normalized params (Hz, dB). Skip for now.
    ".LFOPointModBus",
)

def _is_blacklisted(preset_param: str) -> bool:
    return preset_param.startswith(BLACKLIST_PREFIXES)


def _osc_letter_from_index(idx: int) -> Optional[str]:
    # Serum 2 exposes A/B/C oscillators. Noise/Sub are separate; skip them here.
    return {0: "A", 1: "B", 2: "C"}.get(idx)


def _infer_vst_param_name(preset_param: str) -> Optional[str]:
    """Infer a DawDreamer parameter name from a Serum JSON key.

    Patterns handled:
    - .EnvN.kParamAttack/Decay/Sustain/Release/Hold/Curve{1,2,3}/Start/End
    - .OscillatorN.kParam(Volume|Pan|Coarse|FineTune|Phase|Rand|Retrig|Unison|UnisonDetune|UnisonBlend)
    - .VoiceFilterN.kParam(Freq|Reso|Drive|Var|Keytrack|Type|Slope|Enable)
    - .Global0.kParam(MasterVolume|DirectVol|Pan|Portamento|BendRange|Polyphony)
    - .MacroK.kParamValue
    """
    # Envelopes Env0..Env3 -> Env 1..Env 4
    m = re.match(r"^\.(?:Envelope(?P<e_legacy>[1-4])|Env(?P<e>[0-3]))\.kParam(?P<name>Attack|Decay|Sustain|Release|Hold|Curve1|Curve2|Curve3|Start|End)$", preset_param)
    if m:
        env_idx = int(m.group("e_legacy") or int(m.group("e")) + 1)
        name = m.group("name")
        # Map curves to Atk/Dec/Rel Curve
        curve_map = {"Curve1": "Atk Curve", "Curve2": "Dec Curve", "Curve3": "Rel Curve"}
        if name.startswith("Curve"):
            return f"Env {env_idx} {curve_map[name]}"
        return f"Env {env_idx} {name}"

    # Oscillators Oscillator0..2 -> A/B/C
    m = re.match(r"^\.Oscillator(?P<i>[0-4])\.kParam(?P<name>Enable|Volume|Pan|Coarse|FineTune|Phase|Rand|Retrig|Unison|UnisonDetune|UnisonBlend|Start|End)$", preset_param)
    if m:
        idx = int(m.group("i"))
        name = m.group("name")
        if idx in (0, 1, 2):
            letter = _osc_letter_from_index(idx)
            if letter is None:
                return None
            osc_map = {
                "Enable": "Enable",  # maps to e.g. 'A Enable'
                "Volume": "Level",
                "Pan": "Pan",
                "Coarse": "Semi",
                "FineTune": "Fine",
                "Phase": "Phase",
                "Rand": "Rand Phase",
                # Retrig is ambiguous in exposed params; skip mapping
                "Retrig": None,
                "Unison": "Unison",
                "UnisonDetune": "Uni Detune",
                "UnisonBlend": "Uni Blend",
                "Start": "Start",
                "End": "End",
            }
            suffix = osc_map.get(name)
            if not suffix:
                return None
            return f"{letter} {suffix}"
        elif idx == 3:  # Noise osc
            noise_map = {
                "Enable": "Noise Enable",
                "Volume": "Noise Level",
                "Pan": "Noise Pan",
                "Phase": "Noise Phase",
                "Rand": "Noise Rand Phase",
            }
            return noise_map.get(name)
        elif idx == 4:  # Sub osc
            sub_map = {
                "Enable": "Sub Enable",
                "Volume": "Sub Level",
                "Pan": "Sub Pan",
                "Phase": "Sub Phase",
                "Coarse": "Sub Coarse Pitch",
            }
            return sub_map.get(name)
        return None

    # Voice Filters -> Filter 1/2 ...
    m = re.match(r"^\.VoiceFilter(?P<i>[0-1])\.kParam(?P<name>Freq|Reso|Drive|Var|Keytrack|Type|Slope|Enable)$", preset_param)
    if m:
        idx = int(m.group("i")) + 1
        name = m.group("name")
        filt_map = {
            "Freq": "Freq",      # some plugins expose Cutoff as Freq
            "Reso": "Res",
            "Drive": "Drive",
            "Var": "Var",
            "Keytrack": "Key",
            "Slope": "Slope",
            "Type": "Type",
            "Enable": "Enable",
        }
        suffix = filt_map[name]
        return f"Filter {idx} {suffix}"

    # Global/Master
    m = re.match(r"^\.Global0\.kParam(?P<name>MasterVolume|DirectVol|Portamento|BendRange|Polyphony|FXBus1Vol|FXBus2Vol)$", preset_param)
    if m:
        name = m.group("name")
        if name == "MasterVolume":
            return "Main Vol"
        if name == "DirectVol":
            return "Direct Vol"
        if name == "Portamento":
            return "Porta Time"
        if name == "FXBus1Vol":
            return "Bus 1 Vol"
        if name == "FXBus2Vol":
            return "Bus 2 Vol"
        # BendRange/Polyphony have no direct simple exposed parameter; skip
        return None

    # Macros -> Macro 1..8
    m = re.match(r"^\.Macro(?P<i>[0-7])\.kParamValue$", preset_param)
    if m:
        return f"Macro {int(m.group('i')) + 1}"

    return None


def _normalize_value(preset_param: str, vst_param_name: str, value: float) -> Optional[float]:
    """Normalize raw preset value to [0,1] for DawDreamer.

    We use conservative rules to avoid invalid values. Returns None if we
    cannot confidently normalize the parameter.
    """
    if not isinstance(value, (int, float)):
        return None

    # Already normalized
    if 0.0 <= value <= 1.0:
        return float(value)

    # Helper substrings for percentage-like params
    percent_hints = ("Curve", "Wet", "Width", "Blend", "Mix", "Var", "Key", "Vel", "EnvAmt", "Detune")

    # Envelope time constants (heuristic seconds range)
    if re.search(r"\.Env[0-3]\.kParam(Attack|Decay|Release|Hold)$", preset_param):
        # assume up to 10s typical max
        return max(0.0, min(1.0, float(value) / 10.0))

    if re.search(r"\.Env[0-3]\.kParamSustain$", preset_param):
        # some presets store [0,100]
        return max(0.0, min(1.0, float(value) / 100.0)) if value > 1.0 else float(value)

    if any(h in vst_param_name for h in percent_hints) or any(h in preset_param for h in percent_hints):
        # most of these are 0..100
        if 1.0 < value <= 100.0:
            return float(value) / 100.0

    # Pan: could be -1..1 or -100..100
    if re.search(r"kParamPan$", preset_param) or vst_param_name.endswith(" Pan"):
        v = float(value)
        if -1.0 <= v <= 1.0:
            return (v + 1.0) / 2.0
        if -100.0 <= v <= 100.0:
            return (v + 100.0) / 200.0
        return None

    # Bend Range: typical 2..48
    if "Bend Range" in vst_param_name or re.search(r"\.Global0\.kParamBendRange$", preset_param):
        v = float(value)
        return max(0.0, min(1.0, (v - 2.0) / (48.0 - 2.0)))

    # Unison voices: 1..16
    if re.search(r"\.Oscillator[0-4]\.kParamUnison$", preset_param) or vst_param_name.endswith(" Unison"):
        v = float(value)
        if 1.0 <= v <= 16.0:
            return (v - 1.0) / 15.0
        if 0.0 <= v <= 1.0:
            return v
        return None

    # Master/levels or other simple gains often 0..1 or 0..100
    if vst_param_name in ("Main Vol", "Direct Vol", "Bus 1 Vol", "Bus 2 Vol") or re.search(r"kParamVolume$", preset_param):
        return float(value) / 100.0 if value > 1.0 else max(0.0, min(1.0, float(value)))

    # Porta Time behaves like a time control; cap to 10s range heuristic
    if vst_param_name == "Porta Time":
        v = float(value)
        return max(0.0, min(1.0, v / 10.0 if v > 1.0 else v))

    # Filter Reso/Drive etc often 0..100
    if re.search(r"\.VoiceFilter[0-1]\.kParam(Reso|Drive|Var|Keytrack)$", preset_param):
        return float(value) / 100.0 if value > 1.0 else max(0.0, min(1.0, float(value)))

    # Coarse semitone controls, typical -12..+12
    if (" Semi" in vst_param_name) or ("Coarse Pitch" in vst_param_name):
        v = float(value)
        # try mapping -24..+24 as well, clamp
        if -48.0 <= v <= 48.0:
            # assume +/-12 covers most presets
            return max(0.0, min(1.0, (v + 12.0) / 24.0))
        return None

    # Fine tune often -100..+100 cents
    if vst_param_name.endswith(" Fine"):
        v = float(value)
        if -100.0 <= v <= 100.0:
            return (v + 100.0) / 200.0
        if -1.0 <= v <= 1.0:
            return (v + 1.0) / 2.0
        return None

    # Position/Start/End often 0..100
    if vst_param_name.endswith(" Start") or vst_param_name.endswith(" End"):
        return float(value) / 100.0 if value > 1.0 else max(0.0, min(1.0, float(value)))

    # Frequencies (Hz) and complex FX params: skip to avoid out-of-range glitches
    if re.search(r"(kParamFreq|kParamFrequency)", preset_param) or preset_param.startswith(".FXRack"):
        return None

    # Generic fallbacks
    if 1.0 < value <= 127.0:
        return float(value) / 127.0
    if value > 127.0:
        # too big; likely raw unit (Hz/ms). Skip.
        return None

    # Clamp any remaining value
    return max(0.0, min(1.0, float(value)))


def map_preset_to_vst_parameters(preset_data: Dict[str, float], vst_param_name_to_index: Dict[str, int]):
    """
    Map preset parameters to VST parameters with normalization and safety checks.

    Args:
        preset_data: Dictionary of preset parameters (expect keys from 'dawdreamer_params')
        vst_param_name_to_index: Dictionary mapping VST parameter names to indices

    Returns:
        List of tuples (vst_param_index, value) for successfully mapped parameters
    """
    mapped_params: List[Tuple[int, float]] = []
    skipped_params: List[str] = []

    for preset_param, raw_value in preset_data.items():
        if not isinstance(raw_value, (int, float)):
            continue
        if not preset_param.startswith('.'):
            # we only handle flattened, dotted keys produced by conversion
            continue
        if _is_blacklisted(preset_param):
            continue

        # Resolve a VST param name
        vst_param_name = SERUM2_PARAMETER_MAPPING.get(preset_param) or _infer_vst_param_name(preset_param)
        if not vst_param_name:
            skipped_params.append(preset_param)
            continue

        # Must exist in the plugin
        if vst_param_name not in vst_param_name_to_index:
            skipped_params.append(preset_param)
            continue

        # Normalize value
        norm_value = _normalize_value(preset_param, vst_param_name, float(raw_value))
        if norm_value is None or not (0.0 <= norm_value <= 1.0):
            # Keep noise down but still signal when verbose environments run this
            # print(f"⚠️  Skipping {preset_param} -> {vst_param_name}: cannot normalize {raw_value}")
            skipped_params.append(preset_param)
            continue

        mapped_params.append((vst_param_name_to_index[vst_param_name], norm_value))

    if skipped_params:
        print(f"ℹ️  Skipped {len(skipped_params)} params (unmapped/unsafe). Examples: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}")

    return mapped_params

def get_mapping_coverage(preset_data: Dict[str, float]):
    """
    Get statistics on mapping coverage for a preset.
    
    Args:
        preset_data: Dictionary of preset parameters
    
    Returns:
        Dictionary with mapping statistics
    """
    preset_params = [k for k in preset_data.keys() if k.startswith('.') and not _is_blacklisted(k)]
    # Count explicit + inferable
    explicit = [k for k in preset_params if k in SERUM2_PARAMETER_MAPPING]
    inferable = [k for k in preset_params if k in SERUM2_PARAMETER_MAPPING or _infer_vst_param_name(k)]

    return {
        'total_preset_params': len(preset_params),
        'explicit_mapped_params': len(explicit),
        'inferable_params': len(inferable),
        'coverage_percent_inferable': (len(inferable) / len(preset_params) * 100) if preset_params else 0,
        'unmapped_params': [k for k in preset_params if k not in SERUM2_PARAMETER_MAPPING and not _infer_vst_param_name(k)]
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

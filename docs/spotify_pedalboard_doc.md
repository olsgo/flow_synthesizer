# Spotify Pedalboard Documentation

This document provides a structured overview of [**Spotify Pedalboard**](https://github.com/spotify/pedalboard), a Python library for audio processing. It collects references, issues, and related resources into a coherent instructional format.

---

## Overview

**Pedalboard** is a Python library for working with audio:

- Reading and writing audio.
- Rendering and adding effects.
- Supporting most popular audio file formats.
- Providing built-in audio effects.
- Loading third-party instruments and effects via **VST3®** and **Audio Unit** formats.

Repository: [spotify/pedalboard](https://github.com/spotify/pedalboard)【16†source】

---

## Key Issues & Discussions

### Preset Handling

- [#160](https://github.com/spotify/pedalboard/issues/160) – Request for `.aupreset` support similar to `.vstpreset`.
- [#171](https://github.com/spotify/pedalboard/issues/171) – Add `.aupreset` support.
- Discussion: [JUCE Forum – Add .getPreset / .setPreset](https://forum.juce.com/t/add-getpreset-setpreset-to-audiounitclient/61430/3)【16†source】

### State Management

- [#187](https://github.com/spotify/pedalboard/issues/187) – Save and load presets automatically for VST3 plugins.
- [#289 (comment)](https://github.com/spotify/pedalboard/issues/289#issuecomment) – Explanation of two types of VST3 state (audio vs GUI)【16†source】.
- Relevant JUCE code references:
  - [AudioPluginAudioProcessor get/setStateInformation](https://github.com/juce-framework/JUCE/blob/4f43011b96eb0636104cb3e433894cda98243626/examples/CMake/AudioPlugin/PluginProcessor.h#L42-L43)
  - [VST3PluginInstance get/setStateInformation](https://github.com/juce-framework/JUCE/blob/4f43011b96eb0636104cb3e433894cda98243626/modules/juce_audio_processors/format_types/juce_VST3PluginFormat.cpp#L3055-L3109)
- Tutorial: [JUCE Audio Processor Value Tree State](https://docs.juce.com/master/tutorial_audio_processor_value_tree_state.html)【16†source】

### Preset File Loading

- [#245](https://github.com/spotify/pedalboard/issues/245) – Loading non `.vstpreset` files.
- [#257](https://github.com/spotify/pedalboard/issues/257) – VST plugin program presets not accessible.
- [#266](https://github.com/spotify/pedalboard/issues/266) – Parameter values.
- [#243](https://github.com/spotify/pedalboard/issues/243) – AudioProcessorParameter raw\_value setter issue.

### Additional Feature Requests

- [#211](https://github.com/spotify/pedalboard/issues/211) – ASIO driver support.
- [#240](https://github.com/spotify/pedalboard/issues/240) – LADSPA and LV2 support.
- [#269](https://github.com/spotify/pedalboard/issues/269) – VST2 plugin support.
- [#270](https://github.com/spotify/pedalboard/issues/270) – CLAP plugin support【16†source】.

---

## Machine Learning Integration

- [#263](https://github.com/spotify/pedalboard/issues/263) – ML examples using TensorFlow `tf.data`.
- Example test: [test\_tensorflow.py](https://github.com/spotify/pedalboard/blob/f2c2ccd64e78abaf9b87bc2c59097965c8b92fe5/tests/test_tensorflow.py#L34-L53)【16†source】

---

## Example: Extracting VST3 Plugin State

Helper code for working with `.raw_state` from VST3 plugins (example with Serum):

```python
def is_vst3_xml(raw_state):
    """
    Check if the raw state looks like VST3 XML.
    """
    return (
        len(raw_state) > 13 and
        raw_state[8:13] == b'<?xml' and
        raw_state[-1] == 0x00
    )

def extract_vst3_xml(raw_state):
    """
    Extract the XML content from the VST3 raw state if valid.
    """
    if is_vst3_xml(raw_state):
        try:
            xml_part = raw_state[8:-1].decode('utf-8')
            return xml_part
        except UnicodeDecodeError:
            pass
    return None
```

Full implementation reference: [commit example](https://github.com/0xdevalias/poc-audio-pedalboard/commit/da6a2ae32423fc6d7371d46ebed7273bbc95b45e)【16†source】

Code to write extracted XML: [commit reference](https://github.com/0xdevalias/poc-audio-pedalboard/commit/1e1a4ce95447cbd012df1b5662771be28705892e#diff-88be30bc731c22307bd8ace482b8c8ed449d56fa3ef3eab1551a6e91ffbbd5ccR252-R269)

Example XML output after formatting (from Serum initial state):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<VST3PluginState>
    <IComponent>...</IComponent>
</VST3PluginState>
```

---

## Related Projects & Discussions

- DawDreamer integration example: [discussion](https://github.com/spotify/pedalboard/issues/277#issuecomment)【16†source】
- [#277](https://github.com/spotify/pedalboard/issues/277) – Cannot load `.fxp` file for Serum.

---

## Summary

Spotify Pedalboard is a versatile Python audio library with strong community involvement. Active discussions focus on:

- Expanding plugin/preset support.
- Improving state management.
- Adding ML integration examples.

This document consolidates links and references to assist contributors and users in exploring and extending the library.


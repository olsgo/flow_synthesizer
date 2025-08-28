<!-- 1. serum 2 vst 3 location: '/Library/Audio/Plug-Ins/VST3/Serum2.vst3' -->

1. uad polymax vst 3 location: '/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3'
2. uad polymax factory .vstpreset presets location: '/Library/Audio/Presets/UADx PolyMAX Synth'
3. rendered audio corresponding to each .vstpreset preset (which will form the dataset we will use to train flow synthesizer) should be saved here: '/Users/gjb/Datasets/polymax/render'
4. before trying to handle vst state/preset loading with pedalboard on your own, always reference this example directory so you don't have to start from scratch: '/Users/gjb/Projects/poc-audio-pedalboard'

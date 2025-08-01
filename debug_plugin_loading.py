import dawdreamer as daw
from code.dd_renderer import DDRenderer

# Test Polymax with both methods
engine = daw.RenderEngine(44100, 512)

print("Testing Polymax AU:")
try:
    # Direct method (expected to fail)
    synth = engine.make_plugin_processor("instrument", "/Library/Audio/Plug-Ins/Components/uaudio_polymax.component")
    print(f"Direct method works: {len(synth.get_parameters())} parameters")
except Exception as e:
    print(f"Direct method failed: {e}")

try:
    # DDRenderer method (hoping this works)
    renderer = DDRenderer(44100, 512)
    success = renderer.load_plugin("/Library/Audio/Plug-Ins/Components/uaudio_polymax.component")
    if success:
        print("DDRenderer method works for Polymax!")
    else:
        print("DDRenderer method failed for Polymax: Plugin did not load successfully")
except Exception as e:
    print(f"DDRenderer method failed for Polymax: {e}")

print("\nTesting Polymax VST3:")
try:
    # DDRenderer method with VST3
    renderer2 = DDRenderer(44100, 512)
    success = renderer2.load_plugin("/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3")
    if success:
        print("DDRenderer method works for Polymax VST3!")
    else:
        print("DDRenderer method failed for Polymax VST3: Plugin did not load successfully")
except Exception as e:
    print(f"DDRenderer method failed for Polymax VST3: {e}")
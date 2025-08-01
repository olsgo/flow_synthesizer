# code/dd_renderer.py
import json
import dawdreamer as daw

class DDRenderer:
    def __init__(self, sample_rate=22050, block_size=512):
        self.sr = sample_rate
        self.engine = daw.RenderEngine(self.sr, block_size)
        self.inst = None

    def load_plugin(self, plugin_path: str, name: str = "synth"):
        self.inst = self.engine.make_plugin_processor(name, plugin_path)
        return True

    # --- parameter I/O ---
    def get_parameters_description(self):  # list[dict]
        if self.inst is None:
            return []
        return self.inst.get_parameters_description()

    def get_plugin_parameter_size(self) -> int:
        if self.inst is None:
            return 0
        return self.inst.get_plugin_parameter_size()

    def get_patch(self):
        if self.inst is None:
            return []
        n = self.get_plugin_parameter_size()
        return [(i, self.inst.get_parameter(i)) for i in range(n)]

    def set_patch(self, patch):  # [(index, value_0_1), ...]
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        for idx, val in patch:
            self.inst.set_parameter(int(idx), float(val))

    # --- rendering ---
    def render_patch(self, midi_note=60, velocity=100, note_len_sec=3.0, render_len_sec=4.0):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        self.inst.clear_midi()
        self.inst.add_midi_note(midi_note, velocity, 0.0, note_len_sec)
        self.engine.load_graph([(self.inst, [])])
        self.engine.render(render_len_sec)
        return self.engine.get_audio()  # (channels, samples)

    # optional helpers for presets/state
    def load_vst3_preset(self, preset_path: str):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        return self.inst.load_vst3_preset(preset_path)

    def load_state(self, path: str):
        if self.inst is None:
            raise RuntimeError("No plugin loaded. Call load_plugin() first.")
        return self.inst.load_state(path)
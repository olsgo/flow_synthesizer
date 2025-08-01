import json
import os
from typing import Dict, List, Optional

class PluginConfig:
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file: str = None) -> Dict:
        """Load plugin configuration from file or environment"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration that can be overridden
        return {
            "plugin_path": os.environ.get('SYNTH_PLUGIN_PATH', 'synth/diva.64.so'),
            "plugin_name": os.environ.get('SYNTH_PLUGIN_NAME', 'diva'),
            "params_file": None,  # Auto-generated if None
            "param_defaults_file": None,  # Auto-generated if None
            "selected_params": [],  # Auto-selected if empty
            "max_params": 16,  # Maximum parameters to use for training
        }
    
    def get_plugin_path(self) -> str:
        return self.config['plugin_path']
    
    def get_plugin_name(self) -> str:
        return self.config['plugin_name']
    
    def get_params_file(self) -> str:
        if self.config['params_file']:
            return self.config['params_file']
        return f"synth/{self.get_plugin_name()}_params.txt"
    
    def get_param_defaults_file(self, dataset: str = 'default') -> str:
        if self.config['param_defaults_file']:
            return self.config['param_defaults_file']
        suffix = 'nomod' if dataset == 'toy' else 'default_32'
        return f"synth/{self.get_plugin_name()}_param_{suffix}.json"
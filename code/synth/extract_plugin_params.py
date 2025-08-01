import os
import sys

# Setup macOS library paths before importing librenderman
if sys.platform == 'darwin':
    try:
        from macos_compat import setup_macos_library_paths
        setup_macos_library_paths()
    except ImportError:
        # Fallback for when macos_compat is not in path
        pass

try:
    import librenderman as rm
except ImportError:
    print("Warning: librenderman not available - some functionality will be limited")
    rm = None

import json
from typing import Dict, List, Tuple, Optional
from render_engine_wrapper import create_render_engine, create_patch_generator

class DynamicPluginExtractor:
    def __init__(self, plugin_path: str, config_file="auto_discovery.json"):
        self.plugin_path = plugin_path
        self.config = self._load_config(config_file) if config_file and os.path.exists(config_file) else self._default_config()
        self.engine = None
        self.generator = None
        self.plugin_name = self._extract_plugin_name()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config {config_file}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            "discovery_mode": "auto",
            "auto_select_params": True,
            "max_params": 128,  # Increased from 16 to support more complex synths
            "selection_strategy": "importance_ranking",
            "param_filters": {
                "exclude_keywords": ["bypass", "enable", "on_off", "switch", "midi", "learn"],
                "prefer_keywords": [
                    "macro", "volume", "level", "cutoff", "frequency", "resonance", 
                    "attack", "decay", "sustain", "release", "filter", "osc", "envelope", 
                    "lfo", "tune", "pitch", "mix", "depth", "rate", "feedback", "shape",
                    "fm", "amp", "pan", "delay", "reverb", "chorus", "distortion"
                ]
            },
            "responsiveness_test": {
                "enabled": True,
                "test_range": [0.0, 1.0],
                "min_variance_threshold": 0.01
            }
        }
    
    def _extract_plugin_name(self) -> str:
        """Extract plugin name from file path"""
        basename = os.path.basename(self.plugin_path)
        # Remove common extensions
        for ext in ['.vst3', '.vst', '.dll', '.so', '.dylib']:
            if basename.lower().endswith(ext.lower()):
                basename = basename[:-len(ext)]
        # Clean up the name
        return basename.lower().replace(' ', '_').replace('-', '_')
    
    def discover_parameters(self) -> Dict[int, str]:
        """Dynamically discover all parameters from the plugin"""
        try:
            # Use the wrapper functions instead of direct instantiation
            self.engine = create_render_engine(44100, 512, 512)
            self.engine.load_plugin(self.plugin_path, 0)
            self.generator = create_patch_generator(self.engine)
            
            param_dict = {}
            
            # Method 1: Try to get parameters through random patch generation
            try:
                patch = self.generator.get_random_patch()
                for param_id, value in patch:
                    param_dict[param_id] = f"param_{param_id}"
                print(f"Discovered {len(param_dict)} parameters via random patch method")
            except Exception as e:
                print(f"Random patch method failed: {e}")
            
            # Method 2: Try systematic parameter discovery
            if not param_dict:
                print("Attempting systematic parameter discovery...")
                for i in range(2000):  # Increased range for complex synths
                    try:
                        # Try to set parameter and see if it's valid
                        test_patch = [(i, 0.5)]
                        self.engine.set_patch(test_patch)
                        param_dict[i] = f"param_{i}"
                    except:
                        continue
                    
                    if len(param_dict) >= 500:  # Higher limit for complex synths
                        break
                
                print(f"Discovered {len(param_dict)} parameters via systematic discovery")
            
            # Method 3: Try to get parameter names if RenderMan supports it
            if hasattr(self.engine, 'get_parameter_name'):
                named_count = 0
                for param_id in param_dict.keys():
                    try:
                        name = self.engine.get_parameter_name(param_id)
                        if name and name.strip():
                            param_dict[param_id] = name.strip()
                            named_count += 1
                    except:
                        continue
                print(f"Retrieved names for {named_count} parameters")
            
            return param_dict
            
        except Exception as e:
            print(f"Parameter discovery failed: {e}")
            return {}
    
    def analyze_parameter_importance(self, param_dict: Dict[int, str]) -> List[Tuple[int, str, float]]:
        """Analyze parameter importance based on name patterns and behavior"""
        scored_params = []
        
        for param_id, param_name in param_dict.items():
            score = 0.0
            name_lower = param_name.lower()
            
            # Score based on preferred keywords
            for keyword in self.config["param_filters"]["prefer_keywords"]:
                if keyword in name_lower:
                    score += 10.0
            
            # Penalty for excluded keywords
            for keyword in self.config["param_filters"]["exclude_keywords"]:
                if keyword in name_lower:
                    score -= 15.0
            
            # Bonus for macro controls (common in modern synths like Massive X)
            if "macro" in name_lower:
                score += 20.0
            
            # Bonus for numbered parameters (often important)
            if any(char.isdigit() for char in param_name):
                score += 3.0
            
            # Bonus for common synthesizer sections
            synth_sections = ["osc", "filter", "env", "lfo", "amp", "fx"]
            for section in synth_sections:
                if section in name_lower:
                    score += 8.0
                    break
            
            # Test parameter responsiveness if enabled
            if self.config.get("responsiveness_test", {}).get("enabled", False):
                try:
                    responsiveness = self._test_parameter_responsiveness(param_id)
                    score += responsiveness * 15.0  # Higher weight for responsive params
                except:
                    pass
            
            # Ensure minimum score for any discovered parameter
            score = max(score, 1.0)
            scored_params.append((param_id, param_name, score))
        
        # Sort by score (descending)
        scored_params.sort(key=lambda x: x[2], reverse=True)
        return scored_params
    
    def _test_parameter_responsiveness(self, param_id: int) -> float:
        """Test how much a parameter affects the audio output"""
        if not self.engine:
            return 0.0
        
        try:
            # Render with parameter at minimum
            patch_low = [(param_id, 0.0)]
            self.engine.set_patch(patch_low)
            self.engine.render_patch(60, 100, 1.0, 1.0, True)
            audio_low = self.engine.get_audio_frames()
            
            # Render with parameter at maximum
            patch_high = [(param_id, 1.0)]
            self.engine.set_patch(patch_high)
            self.engine.render_patch(60, 100, 1.0, 1.0, True)
            audio_high = self.engine.get_audio_frames()
            
            # Calculate difference
            if len(audio_low) > 0 and len(audio_high) > 0:
                import numpy as np
                diff = np.mean(np.abs(np.array(audio_high) - np.array(audio_low)))
                return min(diff * 1000, 1.0)  # Normalize to 0-1
            
        except Exception:
            pass
        
        return 0.0
    
    def select_parameters(self, param_dict: Dict[int, str]) -> List[str]:
        """Dynamically select the most important parameters"""
        max_params = self.config.get("max_params", 128)
        
        if self.config.get("selection_strategy") == "importance_ranking":
            scored_params = self.analyze_parameter_importance(param_dict)
            # Filter out parameters with negative scores (excluded keywords)
            valid_params = [(pid, name, score) for pid, name, score in scored_params if score > 0]
            selected = [name for _, name, _ in valid_params[:max_params]]
            
            print(f"Selected {len(selected)} parameters out of {len(param_dict)} discovered")
            if len(selected) > 0:
                print(f"Top parameters: {selected[:10]}")
                
        elif self.config.get("selection_strategy") == "first_n_params":
            selected = list(param_dict.values())[:max_params]
        else:
            # Random selection as fallback
            import random
            param_names = list(param_dict.values())
            selected = random.sample(param_names, min(len(param_names), max_params))
        
        return selected
    
    def generate_config(self) -> Dict:
        """Generate a complete configuration for this plugin"""
        param_dict = self.discover_parameters()
        
        if not param_dict:
            raise Exception(f"Could not discover parameters for {self.plugin_path}")
        
        selected_params = self.select_parameters(param_dict)
        
        config = {
            "plugin_path": self.plugin_path,
            "plugin_name": self.plugin_name,
            "discovered_parameters": {str(k): v for k, v in param_dict.items()},  # Convert keys to strings for JSON
            "selected_parameters": selected_params,
            "total_discovered": len(param_dict),
            "total_selected": len(selected_params),
            "auto_generated": True,
            "generation_config": self.config,
            "generation_timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        return config
    
    def save_config(self, output_path: str = None) -> str:
        """Save the generated configuration"""
        config = self.generate_config()
        
        if not output_path:
            output_path = f"code/synth/configs/{self.plugin_name}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nGenerated configuration for {self.plugin_name}:")
        print(f"  - Discovered {config['total_discovered']} parameters")
        print(f"  - Selected {config['total_selected']} for training")
        print(f"  - Saved to: {output_path}")
        
        # Also save parameter mapping file for compatibility
        param_mapping_path = output_path.replace('.json', '_params.txt')
        with open(param_mapping_path, 'w') as f:
            param_mapping = {v: k for k, v in config['discovered_parameters'].items()}
            f.write(str(param_mapping))
        
        print(f"  - Parameter mapping saved to: {param_mapping_path}")
        
        return output_path
    
    def save_parameter_files(self, dataset_name: str = None) -> Dict[str, str]:
        """Save parameter files in the format expected by the training system"""
        config = self.generate_config()
        
        if not dataset_name:
            dataset_name = f"{self.plugin_name}_{config['total_selected']}par"
        
        # Create parameter list file
        param_list_path = f"code/synth/params/{config['total_selected']}contparams.txt"
        os.makedirs(os.path.dirname(param_list_path), exist_ok=True)
        
        with open(param_list_path, 'w') as f:
            for param in config['selected_parameters']:
                f.write(f"{param}\n")
        
        # Create parameter mapping file
        param_mapping_path = f"code/synth/{self.plugin_name}_params.txt"
        param_mapping = {v: int(k) for k, v in config['discovered_parameters'].items() if v in config['selected_parameters']}
        
        with open(param_mapping_path, 'w') as f:
            f.write(str(param_mapping))
        
        # Create default parameter values file
        param_defaults_path = f"code/synth/param_default_{config['total_selected']}.json"
        param_defaults = {param: 0.5 for param in config['selected_parameters']}  # Default to middle values
        
        with open(param_defaults_path, 'w') as f:
            json.dump(param_defaults, f, indent=2)
        
        files_created = {
            'config': self.save_config(),
            'param_list': param_list_path,
            'param_mapping': param_mapping_path,
            'param_defaults': param_defaults_path
        }
        
        print(f"\nCreated training-ready files:")
        for file_type, path in files_created.items():
            print(f"  - {file_type}: {path}")
        
        return files_created

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dynamically extract and configure VST plugin parameters')
    parser.add_argument('plugin_path', help='Path to VST plugin')
    parser.add_argument('--config', type=str, help='Discovery configuration file (JSON)')
    parser.add_argument('--output', type=str, help='Output configuration file')
    parser.add_argument('--max-params', type=int, help='Maximum number of parameters to select')
    parser.add_argument('--create-training-files', action='store_true', help='Create all files needed for training')
    args = parser.parse_args()
    
    # Load discovery config if provided
    discovery_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            discovery_config = json.load(f)
    
    # Override max_params if specified
    if args.max_params and discovery_config:
        discovery_config['max_params'] = args.max_params
    elif args.max_params:
        discovery_config = {'max_params': args.max_params}
    
    extractor = DynamicPluginExtractor(args.plugin_path, args.config)
    
    if args.create_training_files:
        files = extractor.save_parameter_files()
        print(f"\nTo train with this plugin, use dataset name: {extractor.plugin_name}_{len(extractor.generate_config()['selected_parameters'])}par")
    else:
        config_path = extractor.save_config(args.output)
        print(f"\nTo use this plugin, run:")
        print(f"python osc_launch.py --plugin-config {config_path}")

if __name__ == "__main__":
    main()
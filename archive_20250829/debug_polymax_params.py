import json
from pedalboard import load_plugin

def inspect_polymax_params():
    """Inspect PolyMAX parameters and save schema with details."""
    try:
        plugin = load_plugin('/Library/Audio/Plug-Ins/VST3/uaudio_polymax.vst3')
        
        param_details = {}
        
        for name, param in plugin.parameters.items():
            details = {
                'type': str(type(param)),
                'value': str(getattr(plugin, name)), # Convert value to string
                'min_value': param.min_value,
                'max_value': param.max_value,
            }
            if hasattr(param, 'choices'):
                details['choices'] = param.choices
            
            param_details[name] = details
            
        schema = {
            "plugin_name": "UAD PolyMAX",
            "total_parameters": len(param_details),
            "parameter_details": param_details,
            "schema_version": "3.0",
            "notes": "Detailed parameter schema with types, ranges, and choices"
        }
        
        with open('polymax_param_details.json', 'w') as f:
            json.dump(schema, f, indent=2)
            
        print("Successfully created polymax_param_details.json")
        
    except Exception as e:
        print(f"Failed to inspect parameters: {e}")

if __name__ == '__main__':
    inspect_polymax_params()
from code.dd_renderer import DDRenderer

# Test AU version
renderer = DDRenderer(sample_rate=22050, block_size=512)
try:
    success = renderer.load_plugin("/Library/Audio/Plug-Ins/Components/opsix_native.component")
    print(f"AU Plugin loaded: {success}")
    
    # Get parameter information
    params = renderer.get_parameters_description()
    print(f"Total parameters: {len(params)}")
    for i, p in enumerate(params):
        print(f"  {i}: {p['name']}")
    
except Exception as e:
    print(f"Error loading AU: {e}")

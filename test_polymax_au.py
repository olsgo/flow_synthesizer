from code.dd_renderer import DDRenderer

renderer = DDRenderer(sample_rate=22050, block_size=512)
try:
    success = renderer.load_plugin("/Library/Audio/Plug-Ins/Components/uaudio_polymax.component")
    print(f"Polymax AU loaded: {success}")
    if success:
        params = renderer.get_parameters_description()
        print(f"Total parameters: {len(params)}")
        for i, p in enumerate(params[:20]):
            print(f"  {i}: {p.get('name', 'Unknown')}")
    else:
        print("Plugin path validation:")
        validation = DDRenderer.validate_plugin_path("/Library/Audio/Plug-Ins/Components/uaudio_polymax.component")
        print(f"  {validation['message']}")
        print("Try running: python plugin_debug_tool.py --search uaudio_polymax")
except Exception as e:
    print(f"Error loading Polymax: {e}")
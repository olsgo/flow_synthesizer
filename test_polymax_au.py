from code.dd_renderer import DDRenderer

renderer = DDRenderer(sample_rate=22050, block_size=512)
try:
    ok = renderer.load_plugin("/Library/Audio/Plug-Ins/Components/uaudio_polymax.component")
    print(f"Polymax AU loaded: {ok}")
    if ok:
        params = renderer.get_parameters_description()
        print(f"Total parameters: {len(params)}")
        for i, p in enumerate(params[:20]):
            print(f"  {i}: {p.get('name', 'Unknown')}")
except Exception as e:
    print(f"Error loading Polymax: {e}")
from code.dd_renderer import DDRenderer

renderer = DDRenderer(sample_rate=44100, block_size=512)
try:
    ok = renderer.load_plugin("/Library/Audio/Plug-Ins/VST3/Serum2.vst3")
    print(f"Serum 2 VST3 loaded: {ok}")
    if ok:
        params = renderer.get_parameters_description()
        print(f"Total parameters: {len(params)}")
        print("First 20 parameters:")
        for i, p in enumerate(params[:20]):
            print(f"  {i}: {p.get('name', 'Unknown')}")
        if len(params) > 20:
            print(f"... and {len(params) - 20} more parameters")
except Exception as e:
    print(f"Error loading Serum 2: {e}")
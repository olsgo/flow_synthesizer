from code.dd_renderer import DDRenderer

PLUGIN = "/Library/Audio/Plug-Ins/Components/ZENOLOGY.component"

renderer = DDRenderer(sample_rate=22050, block_size=512)
try:
    ok = renderer.load_plugin(PLUGIN)
    print(f"ZENOLOGY AU loaded: {ok}")
    if ok:
        params = renderer.get_parameters_description()
        print(f"Total parameters: {len(params)}")
        for i, p in enumerate(params[:40]):
            name = p.get('name', f'Param {i}')
            print(f"  {i:03d}: {name}")
except Exception as e:
    print(f"Error loading ZENOLOGY: {e}")


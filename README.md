# YSM Web

Prompt-driven character generation for Lilyco Relay. The model supplies a structured palette, hairstyle, outfit and character-specific details. A Python worker uses [chenx5452321/ysm-ai-engine](https://github.com/chenx5452321/ysm-ai-engine) to sculpt and texture the result, render three views, validate the pack, and package the ZIP. The upstream revision and Beerware notice are in `engine/`.

The ZIP contains a rig with named YSM bones and locators, 29 base motions, a first-person arm model, a 512×512 texture with per-face UVs, and an engine-rendered preview. The output is editable YSM 2.6.0 / Spec 2. Hairstyles and outfits are procedural families. The requested palette and accessories vary per prompt.

## Worker setup

```sh
python3 -m venv engine/.venv
engine/.venv/bin/python -m pip install -r engine/requirements.txt
engine/.venv/bin/python engine/generate.py < engine/example.json > character-ysm.zip
```

For Relay, set `YSM_ENGINE_DIR` to the installed `engine/` directory, or place the worker and virtualenv in `./ysm-engine` relative to the service directory.

Lilyco Relay calls `POST /v1/ysm/generations` with the same bearer key and model selection as chat and returns `application/zip`. In Rust, send `DESIGN_PROMPT` to your text model and await `package_design(&reply)`.

The worker accepts geometry data and fixed preset names. Model replies are never evaluated as code.

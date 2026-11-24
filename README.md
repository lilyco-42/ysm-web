# YSM Web

A Rust library that compiles an AI-designed character into an editable Yes Steve Model Spec 2 ZIP. The caller supplies its own text model. This library runs locally and requires neither a model provider nor credentials.

1. Send `ysm_web::DESIGN_PROMPT` as a system message and the character description as a user message.
2. Pass the model's JSON reply to `ysm_web::package_design`.
3. Serve the resulting bytes as `application/zip`. Import the ZIP in a Spec 2 viewer or unpack it for editing.

The output includes Bedrock main/arm geometry, color-atlas PNG, the YSM manifest, and basic idle/walk/run animation tracks. Each generated cube's position, size and color come from the text-model reply. This is a voxel design workflow with a generated pixel atlas. It does not produce an encrypted proprietary binary or synthesize new motion clips.

Lilyco Relay integrates the library through `POST /v1/ysm/generations` using the same bearer key and text model selection as chat.

```sh
cargo test --locked
```

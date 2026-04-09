# Gateway Native Multimodal Image Passthrough — Implementation Spec

## Goal
Allow Hermes gateway platforms (Telegram, Discord, Slack, etc.) to pass user-attached images directly to the main conversation model when that runtime actually supports native vision, instead of always routing images through the auxiliary vision describer first.

## Why this exists
Today Hermes gateway image handling is lossy, slower, and more expensive than it needs to be:

- `gateway/run.py:2743-2757` always collects image attachments and sends them through `_enrich_message_with_vision(...)`
- `gateway/run.py:5924-5990` always calls `vision_analyze_tool(...)` and prepends a synthetic text description to the user message
- the main model never sees the raw image in the gateway path

That means a screenshot sent to Telegram currently becomes:
1. gateway downloads image
2. auxiliary vision model describes image
3. description text is injected into the user message
4. main model sees only text, not the real image

This is bad for screenshots, charts, code, diagrams, and subtle visual layout questions.

## Important correction to the existing issue
Issue `#5661` is directionally correct, but two details are wrong or incomplete:

1. It says the CLI already does native multimodal image passthrough.
   - Current code does not.
   - `cli.py:2954-3015` also pre-processes attached images through auxiliary vision.
   - The docs currently claim native CLI passthrough, but the code path is still description-first.

2. It treats this as a gateway-only switch.
   - That is only half true.
   - Even if the gateway starts sending multimodal content blocks, `run_agent.py:5203-5218` currently downgrades image blocks to text for `anthropic_messages` before the API call.
   - `agent/anthropic_adapter.py:772-859` already knows how to convert image blocks to Anthropic-native image payloads, so the current unconditional fallback in `run_agent.py` is overly blunt.

So the real feature is not just “skip `_enrich_message_with_vision()`”.
It is “support native multimodal gateway input end-to-end, with conservative fallback when the active runtime cannot actually accept images.”

## Scope

### In scope for this implementation
- Gateway platforms only
- User-attached image inputs (`event.media_urls` where media type is image)
- Native multimodal passthrough for main-model turns
- Conservative fallback to the existing auxiliary description path
- Clean transcript/session persistence without storing giant base64 blobs in session history
- Capability detection for deciding when passthrough is safe
- Tests for gateway, run-agent, and config behavior

### Explicitly out of scope for this implementation
- CLI image-input parity
- Audio/video/omni-modal passthrough
- Changing the `vision_analyze` tool itself
- Making every provider support native images if its transport cannot already do so
- Reworking transcript storage to persist multimodal payloads long-term

CLI parity should be a separate follow-up issue/PR. The current docs can be corrected once the gateway behavior is fixed and the CLI plan is decided.

## Non-negotiable constraints

1. Do not break existing non-vision models.
   - If the active runtime does not support native images, Hermes must keep working through the existing auxiliary description path.

2. Do not persist multimodal content blocks or base64 image payloads into SQLite/JSONL transcripts.
   - `hermes_state.py:857-930` stores `content` as `TEXT` and FTS5 indexes it.
   - Persisting base64 image blocks would bloat session storage and wreck session search.

3. Do not silently send invalid multimodal payloads to unsupported runtimes.
   - Forced passthrough should fail loudly.
   - Auto mode should degrade safely.

4. Do not change the on-demand `vision_analyze` tool behavior.
   - Native passthrough affects the main conversation path only.
   - Tool-driven image analysis remains available.

## Proposed config shape
Add a gateway-specific mode under the existing auxiliary vision section:

```yaml
auxiliary:
  vision:
    gateway_mode: describe   # describe | auto | passthrough
```

### Semantics
- `describe` (default)
  - Current behavior.
  - Always use auxiliary vision pre-description.

- `auto`
  - If the active main runtime is known to support native image input, send real multimodal content blocks.
  - Otherwise fall back to `describe`.

- `passthrough`
  - Force native multimodal input.
  - If the active runtime is known not to support image input, or Hermes cannot determine support safely, fail with a clear error message rather than silently describing the image.

## Why `gateway_mode` instead of a generic `passthrough: true`
The original issue proposed:

```yaml
auxiliary:
  vision:
    passthrough: true
```

That is too vague for Hermes.

Problems with a plain boolean:
- no conservative auto mode
- no way to keep old behavior explicitly
- implies this applies everywhere, even though this PR is gateway-only
- makes failure policy ambiguous

`gateway_mode` is explicit, backward-compatible, and leaves room for a future `cli_mode` or a later unification once both surfaces behave the same way.

## High-level architecture

### Current path
`Gateway event -> _enrich_message_with_vision() -> auxiliary vision model -> synthetic text -> AIAgent.run_conversation(str)`

### New path
`Gateway event -> choose image input mode -> either describe fallback OR build multimodal content blocks -> AIAgent.run_conversation(multimodal content, persist_user_message=text shadow)`

The key design choice is:
- API payload can be multimodal
- persisted transcript must remain clean text

That means the gateway must pass two forms of the same turn:
1. the real API-facing multimodal content
2. a clean text-only shadow string for persistence and resume history

## Proposed implementation by file

### 1. `hermes_cli/config.py`
Add the new default config key:

```python
"auxiliary": {
    "vision": {
        "provider": "auto",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": 30,
        "download_timeout": 30,
        "gateway_mode": "describe",
    },
    ...
}
```

Also validate/document accepted values in the config comments.

### 2. New helper module: `agent/multimodal.py`
Create a shared helper module instead of burying this logic in `gateway/run.py`.

Suggested responsibilities:

- `image_file_to_data_url(path: str | Path) -> str`
  - Convert a cached local image file into a `data:image/...;base64,...` URL
  - Reuse the same MIME-detection logic as `tools/vision_tools.py` rather than duplicating ad hoc logic

- `build_multimodal_user_content(text: str, image_paths: list[str]) -> list[dict]`
  - Create content blocks in OpenAI-compatible format:
    - leading `{ "type": "text", "text": ... }` block when text exists
    - one `{ "type": "image_url", "image_url": { "url": data_url } }` block per image

- `resolve_native_vision_support(provider: str, model: str, api_mode: str, base_url: str = "") -> tuple[bool | None, str]`
  - Return `(supported, reason)` where `supported` is:
    - `True` = safe to use native passthrough
    - `False` = known unsupported
    - `None` = unknown / cannot prove safely

Suggested resolution order:
1. transport guard:
   - allow only `openai_chat`, `codex_responses`, `anthropic_messages`
   - everything else returns `False` or `None`
2. `agent.models_dev.get_model_info(provider, model)`
3. fallback `agent.models_dev.get_model_info_any_provider(model)` for providers like `openai-codex` whose runtime model names map to OpenAI-family models but are not directly in `PROVIDER_TO_MODELS_DEV`
4. conservative fallback:
   - `auto` treats `None` as unsupported and falls back to describe
   - `passthrough` treats `None` as a user-visible error

This helper should be reusable later by the CLI follow-up.

### 3. `gateway/run.py`
Replace the current unconditional enrichment branch with a mode-aware flow.

Current logic:
- `gateway/run.py:2743-2757` collects `image_paths`
- then always calls `_enrich_message_with_vision(...)`

Replace it with a helper flow like:

- gather `image_paths`
- read `auxiliary.vision.gateway_mode`
- inspect the active agent runtime/provider/model
- branch:
  - `describe` -> existing `_enrich_message_with_vision(...)`
  - `auto` + native vision supported -> build multimodal content
  - `auto` + unsupported/unknown -> existing `_enrich_message_with_vision(...)`
  - `passthrough` + supported -> build multimodal content
  - `passthrough` + unsupported/unknown -> fail loudly with a direct user-visible explanation

Important: this code cannot keep assuming `message` is always a string.

Add a small gateway-level helper, something like:
- `_build_gateway_user_input(...) -> tuple[user_payload, persist_text]`

Where:
- `user_payload` is either `str` or `list[dict]`
- `persist_text` is always a plain text string used for transcript/session persistence

The gateway also needs to handle pending model switch notes when `message` is multimodal.
Current code at `gateway/run.py:6869-6874` prepends `_msn` using string concatenation.
That must become:
- if payload is `str`: prepend as today
- if payload is list content blocks: prepend `_msn` as an extra leading text block
- persist shadow text should also include the note

### 4. `run_agent.py`
Broaden the input path so the main agent can accept a multimodal user turn without mangling persistence.

Changes required:

#### Signature / typing
Change:
```python
def run_conversation(self, user_message: str, ...)
```

to something like:
```python
def run_conversation(self, user_message: Any, ..., persist_user_message: Optional[str] = None)
```

or preferably a tighter union type if convenient.

#### Preview/logging safety
Current code assumes `user_message` is a string in at least these places:
- `run_agent.py:6883-6889`
- `run_agent.py:6939`

Add a helper like `_preview_user_message(user_message)` that:
- returns the plain string preview for normal text
- returns a short synthetic preview for list content, e.g.:
  - `"[multimodal user turn: 2 images, 1 text block]"`

#### Persistence safety
The existing `persist_user_message` machinery is exactly what this feature needs.
Keep using it.

The gateway should pass:
- `user_message=<multimodal content blocks>`
- `persist_user_message=<plain text shadow>`

That ensures:
- the API sees the real multimodal payload
- `_apply_persist_user_message_override()` rewrites the in-memory turn before transcript/DB flush
- SQLite/JSONL/FTS stay text-only

### 5. `run_agent.py` anthropic handling
This is the most important “issue text was incomplete” part.

Current behavior:
- `run_agent.py:5203-5218` always converts image content to text for `anthropic_messages`
- but `agent/anthropic_adapter.py:772-859` already supports native image conversion to Anthropic message blocks

That means the current fallback is unconditional when it should be conditional.

Proposed fix:
- keep the existing text-describe fallback path
- only apply it when Hermes has decided native image passthrough is unsafe for the active anthropic-compatible runtime/model
- otherwise pass the original image blocks through to `build_anthropic_kwargs(...)`

Suggested refactor:
- replace the current unconditional `_prepare_anthropic_messages_for_api(...)` behavior with a capability-aware branch
- pseudo-behavior:
  - if no image parts: pass through unchanged
  - if image parts and active runtime supports native vision: pass through unchanged
  - if image parts and active runtime does not support native vision: use the existing image-to-text fallback

This keeps old compatibility behavior available without permanently disabling native Anthropic vision.

### 6. `agent/models_dev.py` or new helper tests
If the current metadata lookup is not enough for `openai-codex`, add the smallest safe fallback needed.

Do not hardcode giant provider/model allowlists in `gateway/run.py`.
Capability resolution belongs in shared model/capability logic.

At minimum the implementation must handle:
- OpenAI Codex models like `gpt-5.4`
- OpenAI chat-compatible providers via `openai_chat`
- Anthropic-native models where native image support is actually available
- conservative fallback for unknown providers/models

### 7. Docs
Update at least one user-facing doc to reflect reality.

Suggested files:
- `website/docs/user-guide/features/vision.md`
- optionally `website/docs/user-guide/features/fallback-providers.md`

Minimum doc updates:
- Gateway can now use native multimodal image passthrough when configured
- `auxiliary.vision.gateway_mode` values and behavior
- Do not claim the CLI already has native passthrough unless the CLI code is fixed in the same PR

## Acceptance criteria

### Functional
1. When a Telegram/Discord/etc. user sends an image and `auxiliary.vision.gateway_mode: describe`, Hermes behaves exactly as it does today.
2. When the user sends an image and `gateway_mode: auto` with a known vision-capable main runtime, Hermes sends a multimodal content block to the main model and does not call auxiliary vision first.
3. When `gateway_mode: auto` and the runtime is not known to support images, Hermes falls back to the current auxiliary description path.
4. When `gateway_mode: passthrough` and the runtime is unsupported or unknown, Hermes returns a clear explanation instead of silently falling back.
5. The persisted transcript/session history stores only plain text shadow content, not multimodal lists or base64 data URLs.
6. `vision_analyze` still works unchanged as an on-demand tool.

### Safety / compatibility
7. Non-image attachments are unaffected.
8. Voice/audio transcription flow is unaffected.
9. Session search and transcript storage are unaffected by giant image payloads.
10. Anthropic-compatible runtimes only receive native image blocks when Hermes has explicitly determined passthrough is safe.

## Test plan

### A. Gateway tests
Add a new focused gateway test module, e.g.:
- `tests/gateway/test_gateway_image_passthrough.py`

Cover:
1. `describe` mode still calls `_enrich_message_with_vision()`
2. `auto` mode with supported runtime builds multimodal content blocks and skips auxiliary vision
3. `auto` mode with unsupported runtime falls back to `_enrich_message_with_vision()`
4. `passthrough` mode with unsupported runtime returns a clear error
5. sender-prefix handling still works in shared threads when payload becomes multimodal
6. pending model-switch note is prepended correctly for multimodal payloads

### B. Agent persistence tests
Add/extend run-agent tests, likely in:
- `tests/run_agent/test_run_agent.py`

Cover:
1. `run_conversation()` accepts list-based user content without crashing preview/logging
2. `persist_user_message` rewrites the current-turn user message before DB/transcript persistence
3. returned message history is text-clean after persistence override

### C. Anthropic routing tests
Add tests proving the new behavior is conditional rather than unconditional:
- native image blocks are preserved for a supported anthropic runtime
- image blocks are converted to text only when fallback is required

This likely belongs in:
- `tests/run_agent/test_run_agent.py`
- or a dedicated anthropic image-path test module if cleaner

### D. Config tests
Add a small config default test if the repo already covers config defaults near this area:
- `gateway_mode` default is `describe`
- invalid values are handled conservatively

## Recommended implementation order

### Phase 1: plumbing and safety
1. Add config default for `auxiliary.vision.gateway_mode`
2. Add shared multimodal helper module
3. Update `run_agent.py` preview/persistence handling so non-string user payloads are safe

### Phase 2: gateway passthrough
4. Add gateway image mode resolver and multimodal payload builder
5. Wire `persist_user_message` shadow text into the gateway `run_conversation(...)` call
6. Handle pending model switch note for multimodal payloads

### Phase 3: anthropic correction
7. Make anthropic image fallback conditional instead of unconditional
8. Add regression tests so native image blocks survive when allowed

### Phase 4: docs and cleanup
9. Update docs for gateway image mode
10. Correct any stale claims that the CLI already does native passthrough, unless the CLI is fixed in the same PR

## What must not be done

1. Do not store multimodal content lists directly in the session DB.
2. Do not add a hardcoded provider:model allowlist in gateway code.
3. Do not silently switch `passthrough` into `describe` mode.
4. Do not expand scope to CLI in the same PR unless the user explicitly wants that.
5. Do not remove `_enrich_message_with_vision()` entirely; it remains the fallback path.

## Follow-up issue after this PR
Create a separate follow-up for CLI parity:
- either bring CLI image input onto the same `gateway_mode`-style native path
- or intentionally keep CLI description-first and fix the docs to match reality

Right now the code and docs disagree. This PR should not pretend otherwise.

## Summary
The correct feature is:
- gateway-native multimodal image passthrough
- capability-aware
- transcript-safe
- fallback-preserving
- not a blind boolean

If implemented this way, Hermes gets the quality and latency win for vision-capable main models without breaking the conservative compatibility story that the current gateway behavior was protecting.

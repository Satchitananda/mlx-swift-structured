# Published release sync — 2026-10-05

The latest published upstream release remains [`mlx-swift-structured` 0.2.0](https://github.com/petrukha-ivan/mlx-swift-structured/releases/tag/0.2.0), commit `b1a1a78f1d61a745f8ba3278c9c99c00316d2c82`. Its tree is identical to the fork's mirrored baseline `6cf6cb8`; `git diff --exit-code 6cf6cb8 refs/upstream-tags/0.2.0` proves the equivalence. Because the fork history was rewritten, the update connects release ancestry with an `ours` merge after that equality check. There is no new upstream source to port or local fix to retire.

The package remains paired with sibling mlx-swift 0.32.3 and mlx-swift-lm 3.32.3. XGrammar stays pinned to `557becfb64c503ae9c04344b0047661f43f44320` (0.2.3), including its required recursive submodules.

## Retained fork fixes

- Runtime compatibility for throwing cache creation, prefill policy, prepared multimodal inputs and iterator state.
- Per-generation matcher composition and observation, including fail-closed mask/accept handling and jump-forward behavior.
- Bounded grammar compiler cache whose eviction preserves active matcher ownership.
- Native error ownership and concurrent error isolation.
- Reserved control-token exclusion from JSON strings, with explicit control grammars still allowed.
- Union of tokenizer, configured and runtime stop-token IDs, with bounds checking.
- Local sibling dependencies and the upstream rejected-tool-call API adaptation in examples.

## Validation and replacement decision

Native SwiftPM build with complete strict concurrency passes on Xcode 27.0 / Swift 6.4. The full suite passes **35 Swift Testing tests in 9 suites**, using freshly compiled sibling Metal kernels and `MLX_ENABLE_TF32=0 swift test --build-system native --skip-build --no-parallel`.

`MLXGuidedGeneration` is present in the LM release and supports the same OS floor. Keep Structured for this release update. A backend migration needs parity for our multiple stop IDs, reserved-token filtering, bounded reasoning matcher, caller sampling, strict compact schema formatting, cache ownership and content-free logs. See the umbrella guided-generation assessment for source evidence and a proposed migration gate.

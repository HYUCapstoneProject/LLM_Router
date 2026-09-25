# Existing-checkpoint context-length qualification

User request: evaluate R0 and R1 without training, then resume the R0 pipeline
with an 8,192-token inference setting if the evidence supports it; qualify R1 too.

## Frozen comparison

- A ordinal (R0), A pointwise (R1), and the subsequently recommended G1 R1-natural
  are separate checkpoint identities. Never substitute one for another.
- Preserve weights, tokenizer, structure-preserving normalization, label order,
  candidate pool, and existing routing decision policy. Change only inference
  maximum length: 512 versus 8,192. Record hashes and library/device versions.
- Keep original training configuration and evaluation results unchanged.
- Reproduce a 512 baseline before interpreting a length effect. Short examples
  must receive identical token IDs; numerical/action differences need inspection.
- Reuse existing validation, sealed-test, and Arena records as diagnostic
  qualification. These are not a new independent test and cannot tune thresholds,
  lambda, the chosen training method, or preprocessing.
- Report entire sets, >512 subsets, lengths/source/category, and paired changes.
  Short examples may reuse verified baseline logits only with explicit provenance;
  the report must distinguish recomputed rows from reused rows.

## Decision rules, fixed before inspecting new predictions

- A runnable 8,192 forward pass alone is not a deployment pass.
- Report paired accuracy/quality changes with uncertainty. Any observed overall
  degradation, source-specific deterioration, increase in severe under-routing,
  or worsened existing cost-budget violation prevents an automatic quality pass
  and requires explaining the tradeoff. Do not silently invent a tolerance after
  seeing the results.
- Small or noisy slices are inconclusive, not evidence of equivalence. Do not
  claim that retained 8,192 architecture proves retained long-input accuracy.
- Runtime qualification needs actual deployment-compatible attention backend,
  device, memory and latency, including a bounded maximum-length case. Production
  isolation and resource headroom come before concurrency/load testing.
- Overall deployment readiness requires both quality and runtime qualification
  plus pre-existing service-pool/promotion requirements. GCP reauthentication is
  currently pending; absence of a runtime measurement cannot be called a pass.

## Conditional implementation

If qualified, keep training max_length=512 as historical fact and record a
separate inference length of 8,192 in a new package/policy. Align candidate gate
evaluation and serving length without changing frozen past experiment configs.
Maintain existing normalization, objective, hash verification, model bounds,
legacy package compatibility, promotion evidence, and rollback. Never edit an
old GCS object or CURRENT solely to make the new contract pass.

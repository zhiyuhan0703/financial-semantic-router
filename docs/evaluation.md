# Evaluation

## Metrics

- `joint_exact`: all three output fields match after canonical business-route ordering.
- `subject_action`: the subject-state decision matches.
- `entity_exact`: the ordered company ID array matches.
- `route_exact`: the canonical route array matches.
- `wrong_binding`: a prediction contains one or more company IDs absent from the expected decision.
- `clarify`: precision, recall, and F1 for the sole `clarify` route.
- module metrics: precision, recall, and F1 for `financial`, `equity`, and `event` independently.

Latency is recorded per case and summarized with P50 and nearest-rank P95 milliseconds. These measurements describe the current machine and provider conditions rather than a service-level guarantee.

Token fields are zero for `pure_rule`. Model routes use provider-reported integer token counts; the scorer stores `null` when the provider does not report usage. Model call rate is total model calls divided by evaluated cases.

## Reproducible public result

The repository commits one offline `pure_rule` TEST report for `public-v1.0`. It can be regenerated without an API key using the command in the root README.

Model-backed routes require a fresh paid call with an explicit model ID, temperature, timeout, retry count, maximum output tokens, run count, and authorization flag. They therefore have no prefilled public score in this repository.

## Reading the comparison

The four routes are compared on the same frozen inputs and nested gold decisions. The experiment is intended to separate identity safety, semantic coverage, verification, latency, and token use. The 24-case TEST split is a compact engineering regression set, not an estimate of production traffic frequency.

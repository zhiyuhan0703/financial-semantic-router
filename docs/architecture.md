# Architecture

## Data flow

```text
query + accepted history + candidate table
  -> deterministic candidate recall
  -> subject state (reuse / new_entity / clarify)
  -> rule shortcut or constrained model decision
  -> shared parser and contract
  -> optional semantic verifier
  -> business routes or clarify
```

## Responsibility boundaries

The candidate table is authoritative for company identity. Candidate recall matches official names, security short names, codes, and configured aliases before any model decision. Clear mentions therefore produce stable IDs deterministically; an ambiguous alias remains a candidate set rather than being guessed.

Subject state is separate from route selection. The router first decides whether the question reuses a confirmed subject, introduces an explicit new entity, or requires clarification. It then selects one or more business routes.

The model receives only `query`, accepted `history`, and company descriptions allowed by the current route. Dataset metadata and the nested `expected` decision never enter a model request.

The four experiment routes share the same output contract:

- `pure_rule` uses candidate recall and deterministic route signals.
- `llm_only` asks the model to produce the complete decision without hard semantic verification.
- `hybrid_no_verifier` fixes deterministic identity information and asks the model only where semantics remain unresolved.
- `hybrid` adds a semantic verifier to the same hybrid decision path.

`hybrid_no_verifier` and `hybrid` therefore differ only in whether the semantic verifier is applied. This makes the verifier's contribution independently measurable.

## Fail-closed behavior

The shared parser first enforces the three-field JSON contract and canonical business-route order. The semantic verifier then rejects unknown IDs, identities outside the allowed candidate set, unsafe changes to a fixed subject, unsupported route shapes, and missing required routes.

When a structured decision is unsafe, fail-closed closes the business path and returns `clarify`. When the provider itself fails or returns unusable content, the route records an explicit error instead of fabricating a model decision.

This repository evaluates level-one routing. Downstream query-parameter extraction, data retrieval, and final-answer generation are separate responsibilities.

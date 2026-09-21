# Financial Semantic Router

A small, auditable routing experiment for multi-turn financial questions. It separates deterministic company identity handling from semantic route selection and compares four routes: pure rules, LLM-only, hybrid without verification, and hybrid with fail-closed verification.

## What it decides

For a current question, up to ten accepted history turns, and a static company candidate table, the router returns:

```json
{"subject_action":"new_entity","company_ids":["SH600036"],"route":["financial"]}
```

- `subject_action`: `reuse`, `new_entity`, or `clarify`
- `company_ids`: stable IDs from the candidate table
- `route`: one or more of `financial`, `equity`, `event`, or the sole fallback `clarify`

`financial` covers financial-statement items, `equity` covers shareholders and ownership, and `event` covers company announcements, penalties, litigation, and other material company events.

## Four experiment routes

| Route | Deterministic candidate recall | LLM decision | Program verifier |
| --- | --- | --- | --- |
| `pure_rule` | yes | no | shared contract only |
| `llm_only` | no hard candidate constraint | yes | no semantic verifier |
| `hybrid_no_verifier` | yes | only for unresolved semantics | no |
| `hybrid` | yes | only for unresolved semantics | yes, fail-closed |

See [docs/architecture.md](docs/architecture.md) for the data flow and boundaries.

## Quick start

Requirements: Python 3.10 or newer. The runtime uses only the Python standard library.

```powershell
python -m unittest discover -s tests -q
python -m scripts.validate_freeze
python -m scripts.run_dev
python -m scripts.run_experiment --route pure_rule --split test --allow-test --output reports/pure_rule-public-v1.0.json
```

All commands above run offline and require no API key.

The default dataset is `frozen/public-v1.0/freeze-manifest.json`: five public A-share company identities plus 8 DEV and 24 TEST routing cases newly authored for this repository.

## Optional model routes

Copy `.env.example` to `.env` or set `DEEPSEEK_API_KEY` in the process environment. The key is read only for model-backed routes. A model run requires every runtime choice to be explicit:

```powershell
python -m scripts.run_experiment --route hybrid --split dev --allow-paid-api --model <provider-model-id> --temperature 0 --timeout-seconds 30 --max-retries 0 --max-tokens 256 --runs 1
```

Replace `<provider-model-id>` at the command line with an ID currently supported by the provider. The repository publishes no copied private model outputs or model accuracy claims.

## Reproducible baseline

The committed report [reports/pure_rule-public-v1.0.json](reports/pure_rule-public-v1.0.json) is produced offline from the public freeze. It records exact match, wrong binding, latency, and call counts. Regenerate it with the command in Quick start.

On the 24-case public synthetic TEST split, the current `pure_rule` baseline is 21/24 joint exact with 3 wrong bindings.

## Data and evaluation

- [DATA_NOTICE.md](DATA_NOTICE.md): origin and reuse boundary
- [docs/data-design.md](docs/data-design.md): case families and gold contract
- [docs/evaluation.md](docs/evaluation.md): metrics and interpretation

## Security

Never commit `.env`, keys, raw provider responses, or private question sets. See [SECURITY.md](SECURITY.md).

## License

Code and newly authored synthetic cases are released under the [MIT License](LICENSE).

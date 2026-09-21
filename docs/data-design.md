# Public Dataset Design

## Scope

`frozen/public-v1.0` contains five public A-share company identities:

- 贵州茅台 (`SH600519`)
- 招商银行 (`SH600036`)
- 中国平安 (`SH601318`)
- 平安银行 (`SZ000001`)
- 美的集团 (`SZ000333`)

`平安` is the only deliberately ambiguous alias. It maps to 中国平安 and 平安银行; an unqualified use must not be bound to either company automatically.

The freeze contains 8 DEV and 24 TEST cases. Every case is marked `source_type: public_synthetic`: the current question, history, family label, and expected decision were newly authored for this repository.

## Case families

| Family | Coverage |
| --- | --- |
| F1 | clear company name, alias, or security code |
| F2 | ambiguous company mention or ambiguous pronoun |
| F3 | recent subject reference and omitted subject |
| F4 | explicit subject switch or exclusion |
| F5 | unknown entity, missing company, or unsupported domain |
| F6 | multiple companies or multiple business routes |
| F7 | long-distance reference and recovery after an unresolved turn |
| F8 | vague intent or route boundary |

## Gold contract

Gold is nested under `expected` and contains exactly:

```json
{
  "subject_action": "reuse | new_entity | clarify",
  "company_ids": ["stable candidate-table IDs"],
  "route": ["financial | equity | event | clarify"]
}
```

Business routes use the canonical order `financial`, `equity`, `event`. `clarify` is a sole route and is never mixed with a business route.

`case_family` and `source_type` are evaluation metadata. `Case.router_input()` exposes only `query`, `history`, and `companies`, so metadata and gold cannot leak into a route decision.

DEV supports implementation diagnostics. TEST is the public offline comparison split and requires the explicit `--allow-test` gate.

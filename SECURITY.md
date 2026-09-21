# Security Policy

## Secrets

Use the process environment or a local root `.env` for `DEEPSEEK_API_KEY`. `.env` and common key-file patterns are ignored; `.env.example` must remain empty. Never paste keys into issues, logs, reports, tests, or example commands.

## Provider failures

Model transport and configuration failures return explicit errors. The router does not invent a fallback model response.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for the repository owner. Do not include live credentials or private datasets in a report.

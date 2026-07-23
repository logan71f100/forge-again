# Security policy

## Supported versions

This is a single-maintainer fork under active development. Security fixes land on
`main` and in the next tagged release; there is no long-term support branch. Always
run the latest `main` or the latest [release](https://github.com/logan71f100/forge-again/releases).

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub: go to the **Security** tab of this repository and
click **Report a vulnerability** (GitHub private security advisories). That keeps the
details confidential until a fix is available.

Please include:

- what the vulnerability allows an attacker to do,
- the steps or a proof-of-concept to reproduce it,
- affected commit or release, OS, and how the server was launched (flags matter).

## Scope

This project inherits most of its code from
[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)
and [AUTOMATIC1111 / stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
A vulnerability that also exists upstream is best reported there as well.

A few deployment notes that are expected behavior, not vulnerabilities:

- The launchers pass `--listen` by default, so the UI and API bind to all interfaces.
  Only run it on a network you trust, or put it behind authentication
  (`--gradio-auth`, `--api-auth`) and/or a reverse proxy. Do not expose it directly
  to the public internet.
- `--api` is enabled by default (the AI assistant uses it). It is unauthenticated
  unless you set `--api-auth`.
- The AI assistant can read and drive the whole UI and run generations by design.
  Only point it at an LLM endpoint you trust.

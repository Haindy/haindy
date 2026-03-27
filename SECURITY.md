# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in HAINDY, please report it responsibly.

**Do not open a public issue.** Instead, email security@haindy.dev with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any suggested fix (optional)

You should receive an acknowledgment within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Scope

HAINDY handles API credentials (stored in the system keychain or encrypted local files) and interacts with OS-level input systems. Security issues in these areas are especially relevant:

- Credential storage and retrieval
- Input injection and automation boundaries
- Session data handling in tool-call mode
- Dependencies with known vulnerabilities

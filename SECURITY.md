# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

The QBioCode team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the project maintainers:

**Primary Contacts**:
- Filippo Utro: futro@us.ibm.com
- Kahn Rhrissorrakrai: krhriss@us.ibm.com
- Aritra Bose: a.bose@ibm.com


**Secondary Contact**:
- Bryan Raubenolt: raubenb@ccf.org

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- You should receive an acknowledgment within 48 hours
- We will send a more detailed response within 7 days indicating the next steps
- We will keep you informed about the progress towards a fix and announcement
- We may ask for additional information or guidance

### Security Update Process

1. The security report is received and assigned to a primary handler
2. The problem is confirmed and a list of affected versions is determined
3. Code is audited to find any similar problems
4. Fixes are prepared for all supported releases
5. New releases are issued and announcements are made

## Security Best Practices for Users

When using QBioCode:

1. **Keep Dependencies Updated**: Regularly update QBioCode and its dependencies
2. **Use Virtual Environments**: Isolate your QBioCode installation
3. **Validate Input Data**: Always validate and sanitize input data
4. **Secure Credentials**: Never commit API keys or credentials to version control
5. **Review Code**: Review any custom code or configurations before deployment
6. **Monitor for Updates**: Watch the repository for security announcements

## Disclosure Policy

- Security vulnerabilities will be disclosed in a coordinated manner
- We aim to fully disclose vulnerabilities within 90 days of the initial report
- Credit will be given to security researchers who report vulnerabilities responsibly

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue.

## Attribution

This security policy is adapted from the [GitHub Security Policy template](https://docs.github.com/en/code-security/getting-started/adding-a-security-policy-to-your-repository).
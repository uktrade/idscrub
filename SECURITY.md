# Security Policy

The Department for Business and Trade (DBT) supports businesses to invest, grow and export, creating jobs and opportunities across the country. Read more about what we do on [GOV.UK](https://www.gov.uk/government/organisations/department-for-business-and-trade/about).

## Reporting a Vulnerability

If you believe you have found a security vulnerability, please submit your report to us using [here](https://hackerone.com/2680e4cd-0436-42a5-bd2a-37fd86367276/embedded_submissions/new)
In your report please include details of:

- The website, IP or page where the vulnerability can be observed.
- A brief description of the type of vulnerability, for example; "XSS vulnerability".
- Steps to reproduce. These should be a benign, non-destructive, proof of concept. This helps to ensure that the report can be triaged quickly and accurately. It also reduces the likelihood of duplicate reports, or malicious exploitation of some vulnerabilities, such as sub-domain takeovers.

Vulnerability reporting guidelines

- Please do not share the vulnerability information beyond the owner and us, without express consent from the owner
- Vulnerabilities reported to the HackerOne platform can be submitted without the need to create a HackerOne account. However, if you wish to be updated you should create an account
- To submit your report, you will need to agree to the HackerOne Terms and Conditions and acknowledge that you have read their Privacy Policy and Disclosure Guidelines
- Once you have submitted the report, it will be assessed by NCC Group within five working days, and forwarded to the affected owners as soon as possible.

The DBT Cyber Team will attempt to make contact with the affected owner. However, the affected owner holds responsibility for resolving the issue.

## Repository security

A new organisation github policy has been created that will enforce a set of security checks we expect a repository in the uktrade account to have. This policy is enabled by default for any new repositories, however existing repositories need to have it applied to them before they can be made public. The expectation is that once enough repositories have been switched from the legacy security policy to the new security policy, this new security policy is enforced across all repositories within the uktrade account

### Custom properties

The uktrade account makes use of custom github properties to enforce branch protection rules and run organisation level github actions. New properties can be added by logging into github using a uktrade account and using the [custom props page](https://github.com/organizations/uktrade/settings/custom-properties)

### Code scanning

All uktrade repositories with the new security policy applied have CodeQL scanning enabled. Individual repositories can apply their own advanced scanning rules if required

### Push protection

To block known secrets being committed into github, all repositories with the new security policy applied will have push protection enabled and enforced.

### Branch protection

An organisation ruleset has been created to apply a minimum set of branch protection rules to each public repository in the uktrade account. These rules are to be seen as the minimum, and repository admins might decide to add additional rules to their own repositories. Organisation admins and repository admins have been added to the bypass list for this branch protection ruleset. The protection rules that will be applied to each repository are:

- A PR is required for merges into the default branch (usually main)
- At least 1 approver is required before a PR can be merged
- Any conversations on the PR must be marked as resolved

As these rules are applied as an organisation ruleset, it is not possible for repository admins to add their own rules that reduce this level of protection. As an example, a repository admin could add a ruleset that drops the required number of approvers to 0 but that would have no effect as the organisation ruleset would take precedence. They could add a ruleset that sets the number of approvers to 3, and as this is not reducing the organisation ruleset protection this would take precedence

# Notes for Gen Contributors

## Code Formatting Guidelines

- 4 spaces per indentation level, no tabs

- try to limit line width to 80 characters

- use ASCII operators and identifiers whenever possible (e.g. use 'theta' instead of 'θ')

## Building docs locally

```sh
./docs/build_docs_locally.sh
```
## Submitting Pull Requests (PRs)

- Include documentation in the same PR as the code.

- Make sure that important info doesn't stay siloed in a PR discussion, since
  users won't see that.

  + If a reviewer asks for clarification about what the code does or is
    intended to do, this often reveals that some info that future users may
    want to know, is missing.  Rather than just respond to the PR discussion
    comment, put that information in the docs.

  + Often, concerns are raised in a PR discussion that lead to a down-scoping
    of the code changes, in order to defer full resolution of the larger issue
    to a later time.  Before merging a PR, make sure any unresolved discussions
    in the PR conversation get moved to new issues.  This way, the unresolved
    discussions do not get lost after the code is merged in.

# Notes for Gen Contributors

## Code Formatting Guidelines

- 4 spaces per indentation level, no tabs

- try to limit line width to 80 characters

- use ASCII operators and identifiers whenever possible (e.g. use 'theta' instead of 'θ')

## Submitting Pull Requests (PRs)

- Include documentation in the same PR as the code.

- Make sure that important info doesn't stay siloed in a PR discussion, since
  users won't see that.

  + If a reviewer asks for clarification about what the code does or is
    intended to do, this often reveals that some info that future users may
    want to know, is missing.  Rather than just respond to the PR discussion
    comment, consider putting that information in the docs.

  + Often, concerns are raised in a PR discussion that lead to a down-scoping
    of the code changes, in order to defer full resolution of the larger issue
    to a later time.  Before merging a PR, make sure any unresolved discussions
    in the PR conversation get moved to new issues.  This way, the unresolved
    discussions do not get lost after the code is merged in.

## Documentation Guidelines

- Write separate introductory material section(s) and an API listing section (titled 'API').

- Write docstrings for API functions. These should be kept short, because more expository material should be included in the introduction section(s) of the documentation, not the API listing.

- In the introduction section(s), provide at least a few sentences of high-level description of the feature, to orient the user. Include external links and internal links. Then, include some short code snippets showing a minimal example of the feature being used. If relevant, include some simple Tex math.

- Build the docs locally to test that it builds and looks right.

- Once you have a made a PR and are assigned a PR `NUMBER`, update your PR description with a link to the automatically generated documentation preview at https://www.gen.dev/docs/previews/PR{NUMBER}.

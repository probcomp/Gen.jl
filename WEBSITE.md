The Gen website is hosted at https://www.gen.dev/.

The website is generated from the source files that are at the head commit of the `gh-pages` branch of the Gen repository.

To edit the website, check out the `gh-pages` branch, and commit and push changes.

Automatically-generated Gen documentation is pushed to the `docs` folder of the `gh-pages` branch whenever a commit is made to the `master` branch of the Gen repository using a combination of Github Actions and Documenter.jl, which are configured in the files `.github/workflows/Documentation.yml` and `docs/make.jl` in the `master` branch. When a pull request is merged into the `master` branch, the `gh-pages` branch is updated with the new documentation, and documentation previews will be cleaned up by the `.github/workflows/Documentation.yml` workflow.

The automatically-managed files and directories are:

- `docs/dev/` (documentation for the version of Gen on the head of `master` branch)

- `docs/v*.*.*/` (directories that contain documentation for each tagged release of Gen)

- `docs/latest/` (a symbolic link to `dev/`)

- `docs/stable/` (a symbolic link to the documentation directory of the latest tagged release Gen)

- `docs/previews/` (documentation previews for unmerged pull requests)

- `docs/versions.js`

*Do not make commits that modify the contents of these automatically managed files and directories.*

To locally test changes to the Gen website, check out the `gh-pages` branch, and follow steps 2 and 4 of [Setting up your GitHub Pages site locally with Jekyll](https://help.github.com/en/articles/setting-up-your-github-pages-site-locally-with-jekyll).

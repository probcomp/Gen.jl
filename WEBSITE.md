The Gen website is hosted on Github pages at https://probcomp.github.io/Gen/

The website is generated from the source files that are at the head commit of the `gh-pages` branch of the Gen repository.

To edit the website, check out the `gh-pages` branch, and commit and push changes.

Automatically-generated Gen documentation is pushed to the `gh-pages` branch whenever a commit is made to the `master` branch of the Gen repository using a combination of Travis and Documenter.jl, which are configured in the files `.travis.yml` and `docs/make.jl` in the `master` branch.
The automatically-managed files and directories are:

- `dev/` (documentation for the version of Gen on the head of `master` branch)

- `v*.*.*/` (directories that contain documentation for each tagged release of Gen)

- `latest/` (a symbolic link to `dev/`)

- `stable/` (a symbolic link to the documentation directory of the latest tagged release Gen)

- `versions.js`

*Do not make commits that modify the contents of these automatically managed directories.*

To locally test changes to the Gen website, check out the `gh-pages` branch, and follow steps 2 and 4 of [Setting up your GitHub Pages site locally with Jekyll](https://help.github.com/en/articles/setting-up-your-github-pages-site-locally-with-jekyll).

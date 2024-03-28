
// We add a simple `onload` hook to inject the custom header for our `HTML`-generated pages
window.onload = function() {
  // <header class="navigation">
  const header = document.createElement('header')
  header.classList.add("navigation")
  header.appendChild((() => {
    const container = document.createElement('div')
    container.classList.add('container')
    container.appendChild((() => {
      const nav = document.createElement('nav')
      nav.classList.add("navbar")

      nav.appendChild((() => {
        const ul = document.createElement("ul")
        ul.classList.add("navbar-nav")

        ul.appendChild((() => {
          const smalllink = document.createElement('li')
          smalllink.classList.add('small-item')
          smalllink.appendChild((() => {
            const a = document.createElement('a')
            a.classList.add("nav-link")
            a.href = 'https://gen.dev'
            a.innerHTML = 'Gen.jl'
            a.title = 'Gen.jl'
            return a
          })())
          return smalllink
        })())

        const items = [
          { title: "Home", link: "https://gen.dev" },
          { title: "Get Started", link: "https://github.com" },
          { title: "Documentation", link: "https://www.gen.dev/docs/stable/" },
          { title: "Examples", link: "https://probcomp.github.io/Gen.jl/" },
          { title: "Papers", link: "http://probcomp.csail.mit.edu" },
          { title: "Contact", link: "http://localhost:1313/" },
          { title: "GitHub", link: "https://github.com/probcomp/", icon: ["fab", "fa-github"] },
        ]

        items.forEach((item) => {
          ul.appendChild(((item) => {
            const li = document.createElement("li")
            li.classList.add("nav-item")
            li.appendChild((() => {
              const a = document.createElement("a")

              if (item.icon !== undefined) {
                a.appendChild((() => {
                  const i = document.createElement("i")
                  i.classList.add(...(item.icon))
                  return i
                })())
              }

              a.classList.add("nav-link")
              a.href = item.link
              a.title = item.title

              a.appendChild((() => {
                const span = document.createElement("span")
                span.innerHTML = `&nbsp;${item.title}`
                return span
              })())

              return a
            })())
            return li
          })(item))
        })
        return ul
      })())
      return nav
    })())
    return container
  })())

  const documenterTarget = document.querySelector('#documenter');

  documenterTarget.parentNode.insertBefore(header, documenterTarget);

  // Edit broken links in the examples, see issue #70
  const editOnGithubLinkTarget = document.querySelector('.docs-edit-link');

  if (editOnGithubLinkTarget) {
    const link = editOnGithubLinkTarget.href;
    if (link.includes('docs/src/examples')) {
      const fixedLink = link.replace('docs/src/', '').replace('.md', '.ipynb');
      editOnGithubLinkTarget.href = fixedLink;
      console.log('Fixed link for the example: ', fixedLink)
    }
  }

}


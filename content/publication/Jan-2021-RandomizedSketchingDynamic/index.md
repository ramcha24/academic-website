---
title: 'Randomized sketching algorithms for low-memory dynamic optimization'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Drew P. Kouri
  - Madeleine Udell

# Author notes (optional)
#author_notes:
#  - 'Equal contribution'
#  - 'Equal contribution'

date: '2021-01-22T00:00:00Z'
doi: '10.1137/19M1272561'

# Schedule page publish date (NOT publication's date).
# publishDate: '2021-01-22T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['2']

# Publication name and optional abbreviated publication name.
publication: In *SIAM Journal on Optimization 2021*
publication_short: In *SIAM-OPT 2021*

abstract: This paper develops a novel limited-memory method to solve dynamic optimization problems. The memory requirements for such problems often present a major obstacle, particularly for problems with PDE constraints such as optimal fow control, full waveform inversion, and optical tomography. In these problems, PDE constraints uniquely determine the state of a physical system for a given control; the goal is to fnd the value of the control that minimizes an objective. While the control is often low dimensional, the state is typically more expensive to store. This paper suggests using randomized matrix approximation to compress the state as it is generated and shows how to use the compressed state to reliably solve the original dynamic optimization problem. Concretely, the compressed state is used to compute approximate gradients and to apply the Hessian to vectors. The approximation error in these quantities is controlled by the target rank of the sketch. This approximate frst- and second-order information can readily be used in any optimization algorithm. As an example, we develop a sketched trust-region method that adaptively chooses the target rank using a posteriori error information and provably converges to a stationary point of the original problem. Numerical experiments with the sketched trust-region method show promising performance on challenging problems such as the optimal control of an advection-reaction-difusion equation and the optimal control of fuid fow past a cylinder.

# Summary. An optional shortened abstract.
summary: Inexact optimization algorithms built on low-rank approximations of the PDE solutions is sufficient to find optimal control vectors. #Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis posuere tellus ac convallis placerat. Proin tincidunt magna sed ex sollicitudin condimentum.

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://par.nsf.gov/servlets/purl/10284903'
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: 'https://drive.google.com/file/d/1AsakZBc2dMINphU7WFE041z25h45_4Of/view?usp=sharing'
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Optimal control of fow past a cylinder.'
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: [] #example
---


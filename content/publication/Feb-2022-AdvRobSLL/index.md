---
title: 'Adversarial robustness of sparse local Lipschitz predictors'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Jeremias Sulam

# Author notes (optional)
#author_notes:
#  - 'Equal contribution'
#  - 'Equal contribution'

date: '2022-02-26T00:00:00Z'
doi: '10.48550/arXiv.2202.13216'

# Schedule page publish date (NOT publication's date).
# publishDate: '2021-01-22T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['2']

# Publication name and optional abbreviated publication name.
publication: In SIAM Journal on Mathematics of Data Science, 2023
publication_short: In *SIMODS 2023*

abstract: This work studies the adversarial robustness of parametric functions composed of a linear predictor and a non-linear representation map. Our analysis relies on sparse local Lipschitzness (SLL), an extension of local Lipschitz continuity that better captures the stability and reduced effective dimensionality of predictors upon local perturbations. SLL functions preserve a certain degree of structure, given by the sparsity pattern in the representation map, and include several popular hypothesis classes, such as piece-wise linear models, Lasso and its variants, and deep feed-forward ReLU networks. We provide a tighter robustness certificate on the minimal energy of an adversarial example, as well as tighter data-dependent non-uniform bounds on the robust generalization error of these predictors. We instantiate these results for the case of deep neural networks and provide numerical evidence that supports our results, shedding new insights into natural regularization strategies to increase the robustness of these models.

# Summary. An optional shortened abstract.
summary: We study the stability of sparse activation patterns of neural networks. An extension of local Lipschitzness that accounts for invariance in activation patterns is provably better for studying robust certification and generalization. 

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://arxiv.org/pdf/2202.13216'
url_code: ''
url_dataset: ''
url_poster: 'https://drive.google.com/file/d/1VpfosWvWXm_263T1eU_ZDxxzqwuRtPl8/view?usp=sharing'
url_project: ''
url_slides: 'https://drive.google.com/file/d/1-ZIiBauikPkE0TAgflBHGGd5hmAjh0iy/view?usp=sharing'
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: 'Hierarchy of Certified Robustness'
  focal_point: 'below'
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


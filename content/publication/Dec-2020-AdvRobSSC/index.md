---
title: 'Adversarial Robustness of Supervised Sparse Coding'

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

date: '2020-12-10T00:00:00Z'
doi: '10.48550/arXiv.2010.12088'

# Schedule page publish date (NOT publication's date).
# publishDate: '2021-01-22T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *Advances in Neural Information Processing Systems 2020*
publication_short: In *NeurIPS 2020*

abstract: Several recent results provide theoretical insights into the phenomena of adversarial examples. Existing results, however, are often limited due to a gap between the simplicity of the models studied and the complexity of those deployed in practice. In this work, we strike a better balance by considering a model that involves learning a representation while at the same time giving a precise generalization bound and a robustness certificate. We focus on the hypothesis class obtained by combining a sparsity-promoting encoder coupled with a linear classifier, and show an interesting interplay between the expressivity and stability of the (supervised) representation map and a notion of margin in the feature space. We bound the robust risk (to `2-bounded perturbations) of hypotheses parameterized by dictionaries that achieve a mild encoder gap on training data. Furthermore, we provide a robustness certificate for end-to-end classification. We demonstrate the applicability of our analysis by computing certified accuracy on real data, and compare with other alternatives for certified robustness.

# Summary. An optional shortened abstract.
#summary: We study the stability of sparse activation patterns of neural networks. An extension of local Lipschitzness that accounts for invariance in activation patterns is provably better for studying robust certification and generalization. 

tags: []

# Display this page in the Featured widget?
featured: false

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'https://proceedings.neurips.cc/paper/2020/file/170f6aa36530c364b77ddf83a84e7351-Paper.pdf'
url_code: 'https://github.com/Sulam-Group/Adversarial-Robust-Supervised-Sparse-Coding'
url_dataset: ''
url_poster: 'https://drive.google.com/file/d/1p-W2PMVtAjuT8BpBz2Hy6vhkjZisX5I1/view?usp=sharing'
url_project: ''
url_slides: 'https://drive.google.com/file/d/1Eeu0UlLaboI_EfSIkTASjBNTnAPXukeL/view?usp=sharing'
url_source: ''
url_video: 'https://slideslive.com/38936682'
url_supplement: 'https://proceedings.neurips.cc/paper/2020/file/170f6aa36530c364b77ddf83a84e7351-Supplemental.pdf'

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


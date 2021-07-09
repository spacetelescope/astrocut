---
title: 'Astrocut: A Python package for astronomical cutouts'
tags:
  - Python
  - astronomy
authors:
  - name: C. E. Brasseur^[corresponding author]
    orcid: 0000-0002-9314-960X
    affiliation: 1
  - name: Carlita Phillip
    affiliation: 1
  - name: Rick White
    affiliation: 1
  - name: Scott Fleming
    orcid: 0000-0003-0556-027X
    affiliation: 1
  - name: Jonathan Hargis
    affiliation: 1
  - name: Susan Mullally
    orcid: 0000-0001-7106-4683
    affiliation: 1
affiliations:
 - name: Space Telescope Science Institute, Baltimore, MD, USA
   index: 1
date: [FILL IN] 2021
bibliography: paper.bib


---

# Summary

Astrocut provides tools for making cutouts from sets of astronomical images with shared footprints. It is under active development. 

Three main areas of functionality are included:

- Solving the specific problem of creating image cutouts from sectors of Transiting Exoplanet Survey Satellite (TESS) full-frame images.
- General fits file cutouts incuding from single images and sets of images with the shared WCS/pixel scale.
- Cutout post-processing functionality, including centering cutouts along a path (for moving targets) and combining cutouts.

# Statement of need

Talk about TESS, talk about fitcut and needing a more easily installable and portable software.

# Design

Talk about cube design and attendend efficiency.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge support from Arfon Smith and Mike Fox during the genesis of this project.

# References
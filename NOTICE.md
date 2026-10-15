# Attribution and scope

Original Python code and configuration use the MIT license in `LICENSE`.
This is an independent educational project; it is not an official Criteo or
Uber product and has no endorsement from them.

## Criteo dataset and derived results

Source: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
The source page specifies **CC BY-NC-SA 4.0**, including attribution,
non-commercial use and share-alike terms. The raw dataset is not distributed
here. The published aggregate tables, figures and data-derived notebook
outputs carry the same [CC BY-NC-SA 4.0 notice](https://creativecommons.org/licenses/by-nc-sa/4.0/).
They are calculations and visualizations created from the v2.1 file and are
not a verbatim copy of the source dataset. The MIT code license does not
grant commercial rights to the dataset or change its terms.

Please cite:

> Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini.
> A Large Scale Benchmark for Uplift Modeling. AdKDD, 2018.

## Methods and prospective observational work

Meta-learners: [Künzel et al.](https://arxiv.org/abs/1706.03461).
Double/debiased machine learning: [Chernozhukov et al.](https://arxiv.org/abs/1701.08687).
DR learning: [Kennedy](https://arxiv.org/abs/2004.14497).
[CausalML](https://github.com/uber/causalml) is a separate Apache-2.0 project.
The IHDP observational extension is a documented future plan, not an executed
experiment or a proprietary Uber dataset. Consult each benchmark source's
own terms when obtaining it.

Third-party package code is not vendored. Learning notes may reference
third-party sources; those sources retain their rights. No blanket MIT
relicensing of third-party text or data is intended.

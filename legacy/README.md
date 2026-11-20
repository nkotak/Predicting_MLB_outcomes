# Archive: the original 2020 project

Everything in `2020_capstone/` is the original student project from May and
June 2020, kept for the record. It is **not maintained and is not used by the
current code**.

What is in here:

- `README_2020.md` and `Proposal.md`: the original write-ups.
- `feature_engineering.ipynb`: the 2020 feature and modelling notebook.
- `graph_EDA.ipynb` and `graphs/`: the 2020 charts.
- `web_scraper/` and `retrosheet_notebooks/`: the 2020 data download and cleaning notebooks.

Things to know before opening the notebooks:

- They do not run as they are. File paths point to one person's laptop, the
  data files they need were never in the repository, and two steps fail on
  current versions of pandas.
- The accuracy figure in `README_2020.md` (58.83%) was the result for the 2019
  season only. The same method averages about 55% across seasons.
- A later add-on package called `grpo_models` (November 2025) was removed. It
  never ran, and it reported about 91% accuracy only because it used
  statistics from the game being predicted. The current package in
  `src/mlb_predict/` replaces both layers.

The current project, its real results and how to run it are described in the
`README.md` at the top of the repository.

# Project page

The static page in `site/` contains the report PDF, four result figures, searchable rankings and the full model-by-task tables. It needs no build step or external JavaScript libraries.

```bash
python3 -m http.server 8771 --bind 127.0.0.1 --directory site
```

Open http://localhost:8771. The files are portable to GitHub Pages under a repository subpath.

| File | Purpose |
|:---|:---|
| `site-config.js` | Repository links, individual weight links, authors and citation |
| `data.js` | All 32 Full-FT and 29 LP configurations, 23 tasks and three seeds |
| `app.js`, `styles.css`, `index.html` | Interactive tables, styling and content |
| `assets/report.pdf` | Current technical report |
| `assets/results*.csv` | Summary, per-task, FT and LP downloads |
| `assets/*.png` | Report figures for the overview, encoder size, data duration and training dynamics |

The model catalog contains the ten default encoders. Rankings and task tables also include Atom-B*, Atom-L* and Atom-H*, each selected by the best evaluated 23-task Full-FT mean. Update the PDF, CSV files and figures together when adding results.

To publish `site/`, select GitHub Actions under repository Settings → Pages, then run the manual **Deploy project page** [workflow](../.github/workflows/pages.yml).

Citation stays hidden while `citationBibtex` in `site-config.js` is empty. Add the arXiv URL and BibTeX after submission.

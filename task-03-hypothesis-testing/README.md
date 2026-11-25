# Task 3: Advanced Statistical Analysis & Hypothesis Testing

**RabTech Academy · Data Science with Python & R Internship**

**Notebook:** [`notebooks/hypothesis_testing.ipynb`](notebooks/hypothesis_testing.ipynb) (executed, all outputs included)

**Question:** does penguin body mass differ by species and sex?
**Data:** Palmer Penguins, n = 333 complete cases (Gorman et al., 2014, CC0), in `data/raw/penguins.csv`.

## What's covered

| Requirement | Notebook section |
|---|---|
| Shapiro-Wilk and Kolmogorov-Smirnov normality tests (plus Lilliefors), Q-Q plots | §3 |
| Two-sample Welch t-tests and Mann-Whitney U, with Holm correction and Cohen's d | §4 |
| One-way ANOVA (plus Welch ANOVA, Kruskal-Wallis, Levene) with Tukey HSD and Games-Howell | §5 |
| Two-way ANOVA (species × sex, Type II) with Tukey HSD across all 6 cells | §6 |
| H₀/H₁ and 95% confidence intervals in markdown | every section |
| Distribution plots and findings | §2, §7 · figures in `reports/figures/` |

## Key findings (α = 0.05)

1. Body mass is normal within every species × sex group (all Shapiro-Wilk and Lilliefors p > 0.05).
2. Males are heavier in all three species: +412 to +805 g, Cohen's d = 1.3–2.7.
3. Species explains about 67% of the variance (η² = 0.67). Gentoo are about 1.4 kg heavier; **Adelie and Chinstrap do not differ** (Tukey p = 0.92).
4. **Significant species × sex interaction** (F = 8.8, p = 0.0002): the sex gap is largest in Gentoo and smallest in Chinstrap.

## Reproduce

```bash
uv sync --locked
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/hypothesis_testing.ipynb
```

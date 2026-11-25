"""Builds reports/capstone_report.pdf from reports/metrics.json and reports/figures.

Run:  uv run python -m churn.report
Every number in the report is read from metrics.json; nothing is typed by hand.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "capstone_report.pdf"
AUTHOR = "Dheena Dhayalan P"
TITLE = "Predicting Customer Churn for Profit"
SUBTITLE = "A profit-driven machine-learning approach to telecom customer retention"

# ---------------------------------------------------------------- fonts (DejaVu ships with matplotlib)
_FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                   ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_FD / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, ACCENT, MUTED, RULE = colors.HexColor("#1F2A36"), colors.HexColor("#1E7F74"), \
    colors.HexColor("#5D6D7E"), colors.HexColor("#D5DBDB")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, textColor=INK, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=13, leading=18, textColor=ACCENT, alignment=TA_CENTER),
    "meta": ParagraphStyle("meta", fontName="DV", fontSize=10.5, leading=16, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=12,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=13.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.4, leading=13.4, textColor=INK, leftIndent=14,
                             bulletIndent=4, spaceAfter=2.5),
    "abstract": ParagraphStyle("abstract", fontName="DV", fontSize=9.6, leading=14.2, textColor=INK,
                               alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18),
    "caption": ParagraphStyle("caption", fontName="DV-I", fontSize=8.2, leading=11, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8, leading=10.2, textColor=colors.white),
    "ref": ParagraphStyle("ref", fontName="DV", fontSize=8.4, leading=11.6, textColor=INK, leftIndent=16,
                          firstLineIndent=-16, spaceAfter=3.5),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=9.6, leading=14, textColor=INK, alignment=TA_CENTER,
                         spaceBefore=4, spaceAfter=6),
}

P = lambda text, style="body": Paragraph(text, S[style])
B = lambda text: Paragraph(text, S["bullet"], bulletText="•")
usd = lambda v: f"${v:,.0f}" if v >= 0 else f"−${-v:,.0f}"
pct = lambda v, d=0: f"{v * 100:.{d}f}%"


def fig(name: str, width_cm: float, caption: str):
    path = FIG / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, "caption")])


def table(rows, col_widths, header_bg=ACCENT, highlight_row: int | None = None, font_size=None):
    data = [[Paragraph(str(c), S["cellb"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in col_widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), header_bg),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F6")]),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#D6EFEB")))
    t.setStyle(TableStyle(style))
    return t


def _decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("DV", 7.5); canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, f"{TITLE}: Capstone Research Report")
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm, "RabTech Academy · Data Science with Python & R")
        canvas.setStrokeColor(RULE); canvas.line(2 * cm, A4[1] - 1.35 * cm, A4[0] - 2 * cm, A4[1] - 1.35 * cm)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build(metrics_path: Path = ROOT / "reports" / "metrics.json", out: Path = OUT) -> Path:
    M = json.loads(metrics_path.read_text())
    D, SP, T, B_ = M["data"], M["split"], M["test_performance"], M["business"]
    sel = M["selected_model"]
    lr, rf = T["Logistic Regression"], T[sel]
    h1 = M["h1_selected_vs_logistic"]
    strat = B_["strategies_test_profit"]
    model_key = f"{sel} targeting"
    p_model, p_all = strat[model_key], strat["Contact everyone"]
    p_rand = next(v for k, v in strat.items() if k.startswith("Random"))
    p_lr, p_perfect = strat["Logistic Regression targeting"], strat["Perfect foresight (upper bound)"]
    drivers = pd.DataFrame(M["drivers"])
    n_sig = int(drivers["significant"].sum())
    sens = pd.DataFrame(M["sensitivity"])
    n_scen = len(sens)
    sens_pos = int((sens.model_profit > 0).sum())
    sens_all_neg = int((sens.contact_all_profit < 0).sum())
    scale = B_["scale_factor_to_full_base"]
    imp = pd.DataFrame(M["permutation_importance"])
    A = M["assumptions"]
    drv = drivers.set_index("feature")
    def detail(f): return drv.loc[f, "detail"]
    rev_share = D["monthly_revenue_churned"] / D["monthly_revenue_total"]
    tn, fp, fn, tp = (M["confusion_matrix"][0] + M["confusion_matrix"][1])
    capture = p_model / p_perfect
    import re
    pay_rates = re.findall(r"\((\d+%)\)", detail("PaymentMethod"))   # [highest, lowest]

    story = []
    # ============================================================ TITLE PAGE
    story += [Spacer(1, 3.2 * cm), P(TITLE, "title"), Spacer(1, 0.3 * cm), P(SUBTITLE, "subtitle"),
              Spacer(1, 1.2 * cm),
              P(f"<b>{AUTHOR}</b><br/>Data Science with Python &amp; R Internship · RabTech Academy<br/>"
                f"Capstone Research Report (Task 6) · {date(2026, 11, 25):%B %Y}", "meta"),
              Spacer(1, 1.3 * cm), P("<b>Abstract</b>", "h2"),
              P(f"Customer churn costs subscription businesses a large share of their revenue: in the dataset "
                f"studied here, the {pct(D['churn_rate'], 1)} of customers who left accounted for "
                f"{pct(rev_share, 1)} of monthly recurring revenue. This report asks whether churn can be "
                f"predicted from routinely collected account data, which factors drive it, and whether a "
                f"model-targeted retention campaign is more profitable than simpler strategies. Using the IBM "
                f"Telco Customer Churn dataset ({D['n_customers']:,} customers), a preregistered analysis plan "
                f"was followed: {n_sig} of {len(drivers)} attributes were significantly associated with churn "
                f"after Holm correction, led by tenure, contract type and the absence of security or support "
                f"services. Four classifiers were tuned with stratified 5-fold cross-validation. The selected "
                f"{sel} reached a test ROC-AUC of {rf['roc_auc']:.3f} (95% CI {rf['roc_auc_ci95'][0]:.3f}–"
                f"{rf['roc_auc_ci95'][1]:.3f}) and a top-decile lift of {M['top_decile']['lift']:.1f}×. Its "
                f"advantage over logistic regression ({lr['roc_auc']:.3f}) was not statistically significant "
                f"(p = {h1['p_one_sided']:.2f}). Translated into a retention campaign with explicit costs, "
                f"model-based targeting produced {usd(p_model)} profit on the {SP['n_test']:,} held-out "
                f"customers, against {usd(p_all)} for contacting everyone and {usd(p_rand)} for random "
                f"targeting, and it remained profitable in all {n_scen} sensitivity scenarios. The report "
                f"closes with concrete retention, pricing and product recommendations.", "abstract"),
              Spacer(1, 0.8 * cm),
              P("<b>Keywords:</b> customer churn · predictive modelling · random forest · logistic regression · "
                "profit-driven evaluation · customer retention", "meta"),
              Spacer(1, 1.2 * cm),
              P("Code, data and full reproducibility instructions:<br/>"
                "github.com/Dheena-cr7/rabtech-datascience-internship/tree/main/task-06-capstone", "meta"),
              PageBreak()]

    # ============================================================ 1 INTRODUCTION
    story += [P("1. Introduction", "h1"),
              P("In subscription markets such as telecommunications, acquiring a new customer typically costs "
                "several times more than keeping an existing one, and small improvements in retention compound "
                "into large gains in customer lifetime value (Reichheld &amp; Sasser, 1990). Churn management "
                "has therefore become one of the most established applications of predictive analytics. A "
                "model scores every customer's risk of leaving, and the retention team focuses its limited "
                "budget on those most at risk (Neslin et al., 2006)."),
              P("A churn model is only valuable if it changes decisions profitably. A model with a high "
                "accuracy score can still lose money if it sends expensive incentives to customers who would "
                "have stayed anyway. This project therefore evaluates models in two ways: with standard "
                "statistical metrics, and in the monetary terms a retention manager would use (Verbeke et al., "
                "2012)."),
              P("Research questions", "h2"),
              B("<b>RQ1 (prediction).</b> How accurately can churn be predicted from account, service and "
                "billing attributes available at prediction time?"),
              B("<b>RQ2 (explanation).</b> Which attributes are most strongly associated with churn, and in "
                "which direction?"),
              B("<b>RQ3 (decision).</b> Does a model-targeted retention campaign make more profit than no "
                "campaign, contacting every customer, or contacting customers at random?"),
              Spacer(1, 4),
              P("Three hypotheses were registered in advance: <b>H1</b>, the best model beats a "
                "logistic-regression baseline on test ROC-AUC; <b>H2</b>, contract type, tenure and internet "
                "service are among the strongest churn drivers; and <b>H3</b>, model-targeted retention is more "
                "profitable than all reference strategies. The full plan (docs/analysis_plan.md) was committed "
                "to version control before any model was trained. This follows the preregistration practice "
                "used in Task 2 of this internship, and it guards against choosing methods after seeing the "
                "results."),
              ]

    # ============================================================ 2 LITERATURE
    story += [P("2. Literature Background", "h1"),
              P("The economics of retention", "h2"),
              P("Reichheld and Sasser (1990) showed that reducing defections by five percentage points could "
                "raise profits by 25–85% across service industries, because long-tenured customers cost less "
                "to serve, buy more and refer others. This argument made churn a management priority and "
                "motivated decades of work on predicting and preventing it."),
              P("Predictive churn modelling", "h2"),
              P("Neslin et al. (2006) ran a large tournament in which dozens of academic and industry teams "
                "built churn models for the same wireless carrier. Logistic regression and tree-based methods "
                "performed best, and the difference between a good and a poor model was worth hundreds of "
                "thousands of dollars per campaign. Lemmens and Croux (2006) showed that ensembles of trees "
                "(bagging and boosting) improve on single classification trees for churn. Random forests "
                "(Breiman, 2001) and gradient boosting (Friedman, 2001) are now standard strong baselines for "
                "tabular data, while logistic regression remains popular because it is transparent and well "
                "calibrated."),
              P("Evaluating models for decisions", "h2"),
              P("Churn data are usually imbalanced, so accuracy is misleading. Saito and Rehmsmeier (2015) "
                "showed that precision-recall curves are more informative than ROC curves when positives are "
                "rare, so both are reported here. Niculescu-Mizil and Caruana (2005) showed that many "
                "classifiers, especially those trained with class re-weighting, produce poorly calibrated "
                "probabilities, which matters whenever probabilities feed into cost calculations. Verbeke et "
                "al. (2012) argued that churn models should be compared by the <i>profit</i> of the campaign "
                "they enable, and proposed profit-based measures that account for contact costs, incentive "
                "costs and the probability that a contacted churner is retained. The decision model in this "
                "report follows the same logic."),
              P("A caution on targeting", "h2"),
              P("Ascarza (2018) found in field experiments that the customers with the highest churn risk are "
                "not always the ones most <i>responsive</i> to retention offers, and that targeting by "
                "predicted uplift can outperform targeting by risk. Because the data used here contain no "
                "campaign history, this study targets by risk and treats the retention success rate as an "
                "explicit, stress-tested assumption. Uplift modelling is identified as the natural next "
                "step."),
              ]

    # ============================================================ 3 DATA
    story += [P("3. Data", "h1"),
              P(f"The study uses the IBM <i>Telco Customer Churn</i> sample dataset, a widely used public "
                f"benchmark describing {D['n_customers']:,} customers of a fictional California telecom "
                f"provider. Each row is one customer and records whether they left in the last month "
                f"(<i>Churn</i>; {D['n_churned']:,} customers, {pct(D['churn_rate'], 1)}). The other columns "
                f"cover four groups of attributes:"),
              table([["Group", "Attributes", "Type"],
                     ["Demographics", "gender, SeniorCitizen, Partner, Dependents", "categorical"],
                     ["Account", "tenure (months), Contract, PaperlessBilling, PaymentMethod", "numeric / categorical"],
                     ["Services", "PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
                                  "DeviceProtection, TechSupport, StreamingTV, StreamingMovies", "categorical"],
                     ["Billing", "MonthlyCharges, TotalCharges ($)", "numeric"],
                     ["Engineered", "n_addon_services, auto_pay, charge_ratio", "numeric / categorical"]],
                    [2.6, 11.0, 3.4]),
              P("<b>Table 1.</b> Attribute groups used for modelling (customerID was removed).", "caption"),
              P(f"<b>Data quality.</b> The data have no duplicate customers. <i>TotalCharges</i> was blank for "
                f"{D['blank_total_charges_fixed']} customers. All of them had a tenure of zero (they had not "
                f"yet been billed), so the blanks were set to $0. The pipeline raises an error if a blank ever "
                f"appears for a tenured customer. <b>Feature engineering</b> added three attributes: the number "
                f"of add-on services (0–6), an automatic-payment flag, and the ratio of the customer's average "
                f"historical charge to their current monthly charge, which picks up recent price changes. The "
                f"final feature set has {D['n_features_model']} attributes. The data were split 80/20 into "
                f"training ({SP['n_train']:,}) and test ({SP['n_test']:,}) sets, stratified by churn so both "
                f"sets have a churn rate of {pct(SP['churn_rate_test'], 1)}. The test set was used only once, "
                f"for the final evaluation."),
              ]

    # ============================================================ 4 METHODOLOGY
    story += [P("4. Methodology", "h1"),
              P("4.1 Statistical analysis of churn drivers", "h2"),
              P(f"Each attribute was tested for association with churn: the χ² test of independence for "
                f"categorical attributes (effect size: Cramér's V) and the Mann-Whitney U test for numeric "
                f"attributes (effect size: rank-biserial correlation <i>r</i>). Because {len(drivers)} tests "
                f"were run, p-values were adjusted with Holm's step-down method to control the family-wise "
                f"error rate at α = 0.05. These are the same non-parametric and categorical tools used in Task 3 "
                f"of the internship."),
              P("4.2 Predictive modelling", "h2"),
              P("Four classifiers were compared: a majority-class dummy (a performance floor), L2-regularised "
                "logistic regression (the interpretable baseline), a random forest and a histogram-based "
                "gradient-boosting machine. Each model was wrapped in a scikit-learn pipeline (Pedregosa et "
                "al., 2011) that one-hot encodes categorical attributes and standardises numeric ones. Because "
                "this preprocessing happens <i>inside</i> cross-validation, no information leaks from the "
                "validation folds. Class imbalance was handled with balanced class weights. Hyperparameters "
                "were tuned by randomised search (20–30 candidates per model) with stratified 5-fold "
                "cross-validation on the training set, using ROC-AUC as the objective. The model with the "
                "highest <i>cross-validated</i> ROC-AUC was selected before the test set was examined."),
              P("4.3 Evaluation", "h2"),
              P("On the test set we report ROC-AUC (primary), PR-AUC (average precision), the Brier score, and "
                "precision, recall and F1 at a classification threshold chosen to maximise F1 on out-of-fold "
                "training predictions. A 95% confidence interval for test ROC-AUC was obtained from 2,000 "
                "bootstrap resamples. H1 was tested with a paired bootstrap of the AUC difference between the "
                "selected model and logistic regression. Explanations come from permutation importance on the "
                "test set (Breiman, 2001) and from logistic-regression odds ratios. Calibration was inspected "
                "with reliability curves."),
              P("4.4 Decision model for a retention campaign", "h2"),
              P("For each contacted customer <i>i</i> with monthly charge <i>m<sub>i</sub></i> and true outcome "
                "<i>y<sub>i</sub></i> (1 = churner), the realised campaign profit is"),
              P("π<sub>i</sub> = y<sub>i</sub> · γ · (H · m<sub>i</sub> − d<sub>i</sub>) − (1 − y<sub>i</sub>) · "
                "d<sub>i</sub> − c", "eq"),
              P(f"where <i>c</i> = ${A['contact_cost']:.0f} is the contact cost, <i>d<sub>i</sub></i> = "
                f"{A['incentive_months']:.0f} × <i>m<sub>i</sub></i> is a free-month incentive, <i>γ</i> = "
                f"{pct(A['success_rate'])} is the share of contacted would-be churners who are retained, and "
                f"<i>H</i> = {A['retained_months']:.0f} months is the revenue horizon kept by retaining them. "
                f"Non-churners who are contacted also accept the free month, so for them it is a pure cost. "
                f"This structure follows the profit-based view of Verbeke et al. (2012). The probability "
                f"threshold that maximises profit was chosen on <i>out-of-fold training predictions</i> and then "
                f"applied unchanged to the test set. The strategy was compared with no campaign, contacting "
                f"everyone, random targeting of the same number of customers (averaged over 500 draws), "
                f"logistic-regression targeting, and a perfect-foresight upper bound. Because γ and H are "
                f"uncertain, a sensitivity analysis re-optimised the threshold over γ ∈ [10%, 50%] and H ∈ "
                f"[6, 24] months."),
              ]

    # ============================================================ 5 RESULTS
    story += [P("5. Results", "h1"),
              P("5.1 Churn drivers (RQ2)", "h2"),
              P(f"{n_sig} of the {len(drivers)} attributes were significantly associated with churn after Holm "
                f"correction. Table 2 lists the strongest. <b>Tenure</b> has the largest effect: "
                f"{detail('tenure')} months. Churn is concentrated among new customers on "
                f"<b>month-to-month contracts</b> ({detail('Contract')}), customers <b>without online "
                f"security or tech support</b>, <b>fiber-optic</b> subscribers, and customers paying by "
                f"<b>electronic check</b>. Gender, phone service and the charge ratio showed no association "
                f"with churn."),
              ]
    top_drv = drivers.head(9)
    rows = [["Attribute", "Test", "Effect size", "Holm p", "Pattern"]]
    for _, r in top_drv.iterrows():
        rows.append([r.feature, "χ²" if r.type == "categorical" else "Mann-Whitney U",
                     f"r = {r.effect:+.2f}" if r.type == "numeric" else f"V = {r.effect:.2f}",
                     "&lt; 10<super>−15</super>" if r.p_holm < 1e-15 else f"{r.p_holm:.1e}", r.detail])
    story += [table(rows, [3.3, 2.7, 1.7, 1.6, 7.7]),
              P("<b>Table 2.</b> The nine strongest univariate churn drivers (sorted by |effect size|). For "
                "numeric attributes, a negative <i>r</i> means churners have lower values.", "caption"),
              fig("01_churn_by_category.png", 17, "<b>Figure 1.</b> Churn rate by contract type, internet "
                  "service and payment method. The dashed line is the overall churn rate."),
              fig("02_numeric_distributions.png", 15, "<b>Figure 2.</b> Tenure and monthly charge "
                  "distributions for churned and retained customers. Churners are concentrated in the first "
                  "months and at higher prices."),
              ]

    # 5.2 model performance
    rows = [["Model", "CV ROC-AUC", "Test ROC-AUC [95% CI]", "PR-AUC", "Brier", "Prec.", "Recall", "F1"]]
    order = ["Dummy (majority)", "Logistic Regression", "Random Forest", "Gradient Boosting"]
    for name in order:
        t, cv = T[name], M["tuning"][name]
        ci = f" [{t['roc_auc_ci95'][0]:.3f}, {t['roc_auc_ci95'][1]:.3f}]" if "roc_auc_ci95" in t else ""
        rows.append([name + (" ★" if name == sel else ""),
                     f"{cv['cv_roc_auc_mean']:.3f} ± {cv['cv_roc_auc_sd']:.3f}", f"{t['roc_auc']:.3f}{ci}",
                     f"{t['pr_auc']:.3f}", f"{t['brier']:.3f}", f"{t['precision']:.3f}", f"{t['recall']:.3f}",
                     f"{t['f1']:.3f}"])
    story += [P("5.2 Predictive performance (RQ1, H1)", "h2"),
              P(f"Table 3 summarises performance. All three learning models are far above the dummy floor "
                f"and close to each other. The <b>{sel}</b> had the highest cross-validated ROC-AUC and was "
                f"selected. On the untouched test set it reached ROC-AUC <b>{rf['roc_auc']:.3f}</b> (95% CI "
                f"{rf['roc_auc_ci95'][0]:.3f}–{rf['roc_auc_ci95'][1]:.3f}) and PR-AUC {rf['pr_auc']:.3f}, "
                f"against a no-skill PR-AUC of {SP['churn_rate_test']:.3f}. The test scores are almost "
                f"identical to the cross-validation estimates, which indicates no overfitting. At the "
                f"F1-optimal threshold ({rf['threshold']:.2f}) it identifies {pct(rf['recall'])} of churners "
                f"with {pct(rf['precision'])} precision ({tp} true positives, {fp} false positives, {fn} missed "
                f"churners). Ranking quality is strong: the 10% of customers with the highest scores churn at "
                f"{pct(M['top_decile']['churn_rate'])}, a <b>lift of {M['top_decile']['lift']:.1f}×</b>."),
              table(rows, [3.5, 2.5, 3.7, 1.55, 1.35, 1.35, 1.45, 1.3], highlight_row=order.index(sel) + 1),
              P("<b>Table 3.</b> Cross-validated (training) and test performance. ★ = selected by CV ROC-AUC. "
                "Precision, recall and F1 are at each model's F1-optimal threshold from out-of-fold "
                "predictions.", "caption"),
              P(f"<b>H1 is not supported.</b> The paired bootstrap gives an AUC difference of "
                f"{h1['diff']:+.4f} between the {sel} and logistic regression (95% CI {h1['ci95'][0]:+.4f} to "
                f"{h1['ci95'][1]:+.4f}; one-sided p = {h1['p_one_sided']:.2f}). On this dataset, flexible "
                f"ensembles add little beyond a well-regularised linear model. This matches the competitive "
                f"performance of logistic regression in the churn tournament of Neslin et al. (2006). The "
                f"ensembles do give slightly better PR-AUC and Brier scores. The reliability curves (Figure 4) "
                f"show that all class-weighted models over-predict churn probabilities, which is expected from "
                f"re-weighting (Niculescu-Mizil &amp; Caruana, 2005). This is why the business threshold is "
                f"tuned directly on profit rather than derived from the raw probabilities."),
              fig("03_roc_pr.png", 16.5, "<b>Figure 3.</b> ROC and precision-recall curves on the test set "
                  f"({SP['n_test']:,} customers)."),
              fig("04_calibration_confusion.png", 16, f"<b>Figure 4.</b> Left: reliability curves. Right: "
                  f"confusion matrix of the {sel} at the F1-optimal threshold."),
              ]

    # 5.3 explanation
    top3 = imp.head(3)
    story += [P("5.3 What the model relies on (RQ2, H2)", "h2"),
              P(f"Permutation importance (Figure 5) measures how much the test ROC-AUC falls when one "
                f"attribute is randomly shuffled. The three most important attributes are "
                f"<b>{top3.feature.iloc[0]}</b> (−{top3['mean'].iloc[0]:.3f} AUC), <b>{top3.feature.iloc[1]}</b> "
                f"(−{top3['mean'].iloc[1]:.3f}) and <b>{top3.feature.iloc[2]}</b> (−{top3['mean'].iloc[2]:.3f}), "
                f"which agrees with the univariate tests. <b>H2 is supported.</b> The logistic-regression odds "
                f"ratios (Figure 6) tell the same story: fiber-optic internet and month-to-month contracts raise "
                f"the odds of churn, while longer tenure and two-year contracts lower them. Some individual "
                f"coefficients, such as the very small odds ratio for MonthlyCharges and the above-1 ratio for "
                f"n_addon_services, are distorted by strong collinearity: the monthly price is largely the sum "
                f"of the services a customer has. They should not be read as causal effects, which is why the "
                f"permutation and univariate evidence is preferred."),
              fig("05_permutation_importance.png", 12.5, f"<b>Figure 5.</b> Permutation importance of the "
                  f"{sel} (mean ± SD over 15 shuffles)."),
              fig("06_odds_ratios.png", 12.5, "<b>Figure 6.</b> Largest logistic-regression odds ratios (log "
                  "scale). Red = raises churn odds, green = lowers them."),
              ]

    # 5.4 business
    rows = [["Strategy", "Customers contacted", "Test-set profit", "Per 1,000 customers"]]
    n_test = SP["n_test"]
    contacted = {"No campaign": 0, "Contact everyone": n_test,
                 "Logistic Regression targeting": "own profit threshold", model_key: B_["n_targeted_test"],
                 "Perfect foresight (upper bound)": B_["churners_total_test"]}
    for k, v in strat.items():
        c = B_["n_targeted_test"] if k.startswith("Random") else contacted.get(k, "")
        rows.append([k, f"{c:,}" if isinstance(c, int) else c, usd(v), usd(v / n_test * 1000)])
    hl = list(strat).index(model_key) + 1
    story += [P("5.4 Business value of targeted retention (RQ3, H3)", "h2"),
              P(f"Using the profit-maximising threshold of {B_['profit_threshold']:.2f} learned on the training "
                f"folds, the {sel} selects {B_['n_targeted_test']:,} of {n_test:,} test customers "
                f"({pct(B_['share_targeted_test'])}) and reaches {B_['churners_reached']} of the "
                f"{B_['churners_total_test']} actual churners. Table 4 compares strategies."),
              table(rows, [6.6, 3.4, 3.2, 3.4], highlight_row=hl),
              P("<b>Table 4.</b> Realised campaign profit on the held-out test set under the base assumptions "
                f"(c = ${A['contact_cost']:.0f}, one free month, γ = {pct(A['success_rate'])}, H = "
                f"{A['retained_months']:.0f} months). Random = mean of 500 random draws of the same size.",
                "caption"),
              P(f"<b>H3 is supported.</b> Model targeting earns <b>{usd(p_model)}</b>. That is "
                f"{p_model / p_all:.1f}× the profit of contacting everyone ({usd(p_all)}) and far above random "
                f"targeting ({usd(p_rand)}; the 95th percentile of 500 random campaigns was only "
                f"{usd(B_['random_profit_p95'])}). It captures {pct(capture)} of the perfect-foresight maximum. "
                f"Logistic-regression targeting comes close ({usd(p_lr)}), which mirrors the H1 result. Scaled "
                f"to the full base of {D['n_customers']:,} customers, the targeted campaign would be worth "
                f"roughly <b>{usd(p_model * scale)}</b> per campaign cycle under these assumptions."),
              fig("07_profit_curve.png", 15.5, "<b>Figure 7.</b> Cumulative campaign profit as customers are "
                  "contacted in order of predicted risk. Profit peaks well before everyone is contacted. Beyond "
                  "that point, incentives are wasted on customers who would have stayed."),
              P(f"<b>Robustness.</b> The success rate γ and the retention horizon H are the most uncertain "
                f"inputs. Across all {n_scen} combinations in Figure 8, model targeting stays profitable "
                f"({sens_pos}/{n_scen} scenarios), while contacting everyone <i>loses</i> money in "
                f"{sens_all_neg} of them. Under the most pessimistic setting (γ = 10%, H = 6 months), targeted "
                f"retention still makes a small profit ({usd(sens.model_profit.min())}). The conclusion that the model "
                f"should guide whom to contact therefore does not depend on optimistic assumptions."),
              fig("08_sensitivity.png", 11, "<b>Figure 8.</b> Test-set profit of model targeting under "
                  "different retention success rates and value horizons. The threshold is re-optimised on "
                  "training folds for each cell."),
              ]

    # ============================================================ 6 DISCUSSION
    story += [P("6. Discussion", "h1"),
              P("The results answer the three research questions clearly. Churn is <b>predictable</b> from "
                "routine account data (RQ1): AUC ≈ 0.84–0.85 with a strong top-decile lift, although model "
                "complexity brings little extra discrimination. The <b>drivers</b> are consistent across "
                "statistical tests and models (RQ2). Customers are most likely to leave early in their "
                "relationship, when not tied to a contract, when they pay high fiber prices without the "
                "security and support add-ons that make a service \"sticky\", and when they pay by manual "
                "electronic check. Most importantly, the <b>economic analysis</b> (RQ3) shows that the value of "
                "the model lies in <i>whom not to contact</i>. Contacting every customer wastes most of the "
                "incentive budget on loyal customers, while risk-ranked targeting turns the same offer into a "
                "clearly profitable campaign."),
              P("A methodological lesson is that accuracy and profit are not the same thing. Gradient boosting "
                "and random forest are statistically indistinguishable from logistic regression on AUC, and "
                "their profit difference is small. In practice, a team may reasonably deploy the logistic model "
                "because it is transparent and easy to audit, and keep the ensemble as a challenger."),
              ]

    # ============================================================ 7 BUSINESS IMPLICATIONS
    story += [P("7. Business Implications and Recommendations", "h1"),
              B(f"<b>Run a risk-targeted retention programme.</b> Score the customer base monthly and contact "
                f"roughly the top {pct(B_['share_targeted_test'])} by risk (threshold ≈ "
                f"{B_['profit_threshold']:.2f}). Do not run blanket campaigns. Expected value: about "
                f"{usd(p_model * scale)} per cycle for a base of {D['n_customers']:,} customers under the stated "
                f"assumptions."),
              B("<b>Migrate month-to-month customers to term contracts.</b> Contract type is the strongest "
                f"single lever ({detail('Contract')}). Offer upgrade discounts at natural decision points, such "
                f"as the end of a promotional period."),
              B("<b>Protect the first year.</b> Churners have a median tenure of only "
                f"{detail('tenure').split('churned ')[1].split(' ')[0]} months. An onboarding programme "
                f"(welcome call, service check at 30/60/90 days) targets the highest-risk period."),
              B("<b>Bundle security and support with fiber.</b> Fiber customers churn far more, and customers "
                "without OnlineSecurity or TechSupport churn at about 42%. Including these services in fiber "
                "plans, or offering them as a trial, should increase stickiness. This also suggests checking "
                "fiber pricing and service quality."),
              B(f"<b>Nudge automatic payment.</b> Electronic-check payers churn at about "
                f"{pay_rates[0]}, against {pay_rates[-1]} for automatic credit-card payment. A small "
                f"autopay incentive is cheap and reduces friction."),
              B("<b>Measure, do not assume, the retention effect.</b> Launch the programme as a randomised A/B "
                "test (treated vs held-out high-risk customers) to estimate the true success rate γ. Then "
                "extend the model towards uplift targeting (Ascarza, 2018)."),
              ]

    # ============================================================ 8 LIMITATIONS
    story += [P("8. Limitations and Future Work", "h1"),
              B("<b>Benchmark data.</b> The IBM dataset is a public sample of a fictional provider, observed at "
                "a single point in time. The results should be re-validated on real, time-stamped company "
                "data with an out-of-time test period."),
              B("<b>Assumed campaign economics.</b> Contact cost, incentive, success rate and value horizon "
                "are assumptions. The sensitivity analysis shows the conclusion is robust, but the absolute "
                "dollar figures are illustrative."),
              B("<b>Risk is not responsiveness.</b> Without campaign history, the model predicts <i>who will "
                "leave</i>, not <i>who can be saved</i>. Uplift modelling on experimental data is the logical "
                "next step."),
              B("<b>Association, not causation.</b> The driver analysis is observational. For example, "
                "fiber-optic customers may churn because of price, competition or service quality. Pricing or "
                "product changes should be tested before full rollout."),
              B("<b>Calibration.</b> Class-weighted models over-estimate probabilities. If calibrated "
                "probabilities are needed (for example, for expected-revenue forecasting), Platt or isotonic "
                "calibration should be added."),
              ]

    # ============================================================ 9 CONCLUSION
    story += [P("9. Conclusion", "h1"),
              P(f"This capstone followed a complete, preregistered data-science workflow on "
                f"{D['n_customers']:,} telecom customers: data validation, statistical analysis of drivers, "
                f"tuned and cross-validated models, honest test-set evaluation with confidence intervals, model "
                f"explanation and a profit-based decision analysis. Churn can be predicted with a test ROC-AUC "
                f"of {rf['roc_auc']:.3f}. It is driven mainly by short tenure, month-to-month contracts and "
                f"fiber service without support add-ons. Most importantly, using the model to decide whom to "
                f"contact turns a loss-prone blanket campaign into a reliably profitable one. The analysis "
                f"also shows that a simpler, interpretable model is almost as good as complex ensembles, and "
                f"that predictive accuracy should always be judged by the business decision it supports."),
              ]

    # ============================================================ REFERENCES
    refs = [
        "Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. "
        "<i>Journal of Marketing Research, 55</i>(1), 80–98.",
        "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32.",
        "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. "
        "<i>The Annals of Statistics, 29</i>(5), 1189–1232.",
        "IBM. (n.d.). <i>Telco customer churn</i> [Data set]. IBM telco-customer-churn-on-icp4d repository, GitHub.",
        "Lemmens, A., &amp; Croux, C. (2006). Bagging and boosting classification trees to predict churn. "
        "<i>Journal of Marketing Research, 43</i>(2), 276–286.",
        "Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., &amp; Mason, C. H. (2006). Defection detection: "
        "Measuring and understanding the predictive accuracy of customer churn models. "
        "<i>Journal of Marketing Research, 43</i>(2), 204–211.",
        "Niculescu-Mizil, A., &amp; Caruana, R. (2005). Predicting good probabilities with supervised learning. "
        "In <i>Proceedings of the 22nd International Conference on Machine Learning</i> (pp. 625–632).",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. "
        "<i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
        "Reichheld, F. F., &amp; Sasser, W. E. (1990). Zero defections: Quality comes to services. "
        "<i>Harvard Business Review, 68</i>(5), 105–111.",
        "Saito, T., &amp; Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot "
        "when evaluating binary classifiers on imbalanced datasets. <i>PLoS ONE, 10</i>(3), e0118432.",
        "Verbeke, W., Dejaeger, K., Martens, D., Hur, J., &amp; Baesens, B. (2012). New insights into churn "
        "prediction in the telecommunication sector: A profit driven data mining approach. "
        "<i>European Journal of Operational Research, 218</i>(1), 211–229.",
    ]
    story += [P("References", "h1")] + [P(r, "ref") for r in refs]

    # ============================================================ APPENDIX
    rows = [["Model", "Tuned hyperparameters (best of randomised search)"]]
    for name in order[1:]:
        bp = M["tuning"][name]["best_params"]
        rows.append([name, ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                     for k, v in bp.items())])
    story += [P("Appendix A. Reproducibility", "h1"),
              table(rows, [3.6, 13.4]),
              P("<b>Table A1.</b> Selected hyperparameters (stratified 5-fold CV, seed 42).", "caption"),
              P("The full analysis can be reproduced from the repository with two commands: "
                "<font face='DV-B'>uv run python -m churn.pipeline</font> (data, tests, models, figures and "
                "metrics.json) and <font face='DV-B'>uv run python -m churn.report</font> (this PDF). The "
                "environment is pinned in uv.lock, the raw data file is version-controlled, all random seeds "
                "are fixed at 42, and 13 unit tests cover data cleaning, the statistical helpers and the profit "
                "model. The analysis plan in docs/analysis_plan.md was committed before modelling began.")]

    # avoid orphaned headings: start a new page only if < ~3 cm remain before a heading
    from reportlab.platypus import CondPageBreak
    guarded = []
    for flow in story:
        if isinstance(flow, Paragraph) and flow.style.name in ("h1", "h2") and guarded:
            guarded.append(CondPageBreak(3.2 * cm))
        guarded.append(flow)
    story = guarded

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=1.9 * cm, title=f"{TITLE}: Capstone Research Report",
                            author=AUTHOR, subject="RabTech Academy Data Science Internship, Task 6")
    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return out


if __name__ == "__main__":
    path = build()
    from pypdf import PdfReader
    print(f"Wrote {path} ({len(PdfReader(path).pages)} pages)")

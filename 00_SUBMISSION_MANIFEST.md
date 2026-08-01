# Submission Package — Scientific Reports

**Manuscript:** *A Multi-Model, Multi-Domain Benchmark of Large Language Model Agreement with Humans and with Each Other in Sentiment Classification*
**Author:** Aneesh K Sajan (sole) · Independent Researcher, Seattle, WA, USA · asajannow@gmail.com · ORCID 0009-0002-5704-957X
**Journal:** *Scientific Reports* (Nature Portfolio) — journal code 41598
**Portal:** SNAPP submission `80ce4765-70a1-4a89-bcf1-104831c6c65d`
**Package prepared:** 19 Jul 2026 · Manuscript source version: v15 (v2 edits + Gemini claim scoped to Batch-API pathway)

---

## Files in this package → SNAPP upload slot

**READY-TO-UPLOAD LaTeX + PDF are now built** (xelatex, 0 errors). Upload the `.zip` files.

| File | SNAPP "Files" slot | Notes |
|---|---|---|
| **`manuscript_latex.zip`** | **Manuscript file** | Self-contained `manuscript.tex` + 3 figures. Compiles with **xelatex** → 18 pp. **Upload this.** |
| **`supplementary_latex.zip`** | **Supplementary material** | Self-contained `supplementary.tex` + 2 figures → 11 pp PDF. **Upload this.** |
| **`03_Cover_Letter.md`** | **Cover letter** | Add the date + reviewer emails; paste as text or PDF. |
| `manuscript.pdf` / `supplementary.pdf` | (preview / PDF-format alternative) | Compiled previews; PDF is also an accepted first-submission format. |
| `manuscript.tex` / `supplementary.tex` | (loose sources, same as in zips) | For editing. |
| `01_Manuscript.md` / `02_Supplementary_Information.md` | (markdown sources) | Editable source of truth. |
| `figures/` | (embedded; separate hi-res only if accepted) | Figures 1–3, S1–S2; filenames match numbers. |

---

## What you still need to do before uploading

**1. Cover letter.** Add today's date and current emails for the three suggested reviewers (or replace them).

**2. Portal form fields (SNAPP tabs)** — mirror the manuscript:
- *Details*: title, abstract (≤200 words), the 6 keywords.
- *Authors*: Aneesh K Sajan, Independent Researcher, Seattle WA USA, asajannow@gmail.com, ORCID 0009-0002-5704-957X (corresponding).
- *Declarations*: Competing interests = "The author declares no competing interests."; Funding = none; Data availability = repo URL; Ethics = not applicable (public datasets); AI use = documented in Methods.

---

## Compliance checklist (verified against Sci Rep guidelines)

- [x] Title ≤ 20 words, single declarative sentence (18 words)
- [⚠] Abstract = **225 words** — OVER the 200 recommendation (kept per author's exact wording; trim before submission if the portal enforces 200)
- [x] Up to 6 keywords
- [⚠] Main text = **4,711 words** — OVER the 4,500 recommendation (author's wording kept; +~90 from the Gemini pathway-confound scoping; soft limit)
- [x] IMRaD: Introduction → Results (subheadings) → Discussion (no subheadings) → Methods
- [x] Unnumbered headings; no footnotes
- [x] ≤ 8 display items in main text (5 tables + 3 figures); overflow in SI (18 tables + 2 figs + 1 note)
- [x] Title page: author, affiliation, corresponding author marked `*`, ORCID
- [x] Mandatory end-matter: Acknowledgements, Author Contributions, Data Availability, Competing Interests, Funding, Ethics, LLM-use (in Methods)
- [x] References: 60, Nature style, `[n]` in text, all authors verified (phantom [1] replaced; 8 author/title errors corrected)
- [x] Data availability names a public repository (no "on request"); reviewer access during review
- [x] Statistical reporting: exact P values, Bonferroni α, bootstrap CIs (incl. pairwise-κ CIs), labelled n
- [x] SI: single file, title+author on page 1, items numbered S1…, referenced from main text
- [x] **LaTeX built + compiles cleanly** (xelatex, 0 errors): manuscript 18 pp, SI 11 pp; zips ready
- [x] Data/code repository is public: https://github.com/aneeshks/Multi-Domain-Sentiment-Benchmark
- [ ] **Cover letter date + reviewer emails** (your step)

---

## Second-round peer-review items — all addressed in this version

M1 (FinBERT contradiction) → resolved (FinBERT removed, RoBERTa sole supervised baseline).
M2 (Gemini artifact-proofing) → invalid-rate table (Gemini = 0), confusion matrices (Supp. Table S17), Supp. Note S1; raw-output limitation disclosed.
M3 (repro package to reviewers) → Data Availability reworded for review-time access.
m1–m10 → all applied (API-call count, RoBERTa-leakage caveat, S8/S9 refs, pairwise-κ CIs, ref verification, PII wording, licence note, figure filenames, three-class-only mean, Gemini subset labels).

*Provenance: manuscript `sr_manuscript_source_v15.md`; SI `sr_supplementary_information_v3.md`; reviewer response `REVIEWER_RESPONSE.md`; requirements `00_SCIENTIFIC_REPORTS_REQUIREMENTS.md`; analysis scripts + metrics under `substudy_results/`.*

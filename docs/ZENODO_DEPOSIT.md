# GitHub and Zenodo deposit

The recommended route is a public GitHub repository archived through Zenodo.
This supplies a browsable code repository and a permanent, versioned software
DOI from the same release.

## Metadata of record

- **Repository name:** `gb-day-ahead-probabilistic-forecasting`
- **Release tag/version:** `v1.0.0`
- **Resource type:** Software
- **Access:** Open
- **Licence:** MIT
- **Software creator:** Anirudh Agrawal, UCL Energy Institute, University
  College London
- **Contributor:** Aidan O'Sullivan, UCL Energy Institute, University College
  London; role: Supervisor
- **Title:** Probabilistic day-ahead electricity-price forecasting in Great
  Britain: modelling and evaluation code
- **Language:** English; implementation language: Python
- **Keywords:** probabilistic forecasting; electricity price forecasting;
  quantile regression averaging; conformal prediction; day-ahead electricity
  market; machine learning

Add ORCIDs to the GitHub/Zenodo record if the authors wish to provide them. Do
not invent identifiers or use the journal article DOI as the software DOI.

## Deposit sequence

1. Run the final local audit:

   ```bash
   python run_pipeline.py audit-release
   ```

2. Create a public GitHub repository using the repository name above. Add the
   **contents** of `code_release/` at its root, including `LICENSE`,
   `CITATION.cff` and `.gitignore`. Do not add local data, forecast artefacts,
   caches or environments.
3. Check the GitHub repository's licence and citation panels. They should read
   the MIT licence and `CITATION.cff` automatically.
4. Sign in to Zenodo, connect the GitHub account, select **GitHub**, sync the
   repository list and enable this repository.
5. Create the GitHub release `v1.0.0`. Zenodo will ingest the release and mint
   a version-specific DOI. Confirm that the Zenodo record lists the resource
   as Software, Anirudh Agrawal as creator, Aidan O'Sullivan as supervisory
   contributor, and MIT as the licence before final publication.
6. Record both the version DOI and the concept DOI shown by Zenodo. Use the
   version DOI when citing the exact code supporting the paper; the concept DOI
   can identify the evolving software across releases.
7. Add the public repository URL and version DOI to the manuscript Data
   Availability statement. The journal article DOI, once assigned, can be
   added to the Zenodo record as a related work using the relation
   `isSupplementTo`.

Zenodo publication is an external, effectively permanent action. Preview all
files and metadata before publishing the record.

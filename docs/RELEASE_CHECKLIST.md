# Public-release checklist

## Complete locally

- Code-only repository structure created.
- Licensed EPEX data, hourly realised prices and environments excluded.
- Public-source access dates documented.
- Manuscript LEAR-QRA and three-model QRA scripts included.
- Aggregate, non-disclosive reference result tables included.
- Local snapshot preparation and release-audit scripts included.
- MIT software licence added, with Anirudh Agrawal as copyright holder.
- `CITATION.cff` added, with Anirudh Agrawal as the software creator.
- GitHub and Zenodo deposit metadata and sequence documented.

## Required before publication

- Add creator and contributor ORCIDs if they are to be included; do not invent
  or infer them.
- Create the public repository and run `python run_pipeline.py audit-release`
  on the material to be committed.
- Tag the exact archived version (suggested: `v1.0.0`).
- Connect the repository to Zenodo, create the archived release and record its
  version and concept DOIs. Record Aidan O'Sullivan as a supervisory
  contributor, not as a software creator.
- Replace the manuscript Data Availability TODO with the repository citation,
  DOI and source access dates.
- Create the separate journal highlights file and set `\showtodosfalse` for the
  submission build.

Creating a GitHub repository or Zenodo record changes external state and is not
performed by the local build scripts.

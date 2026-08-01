# Protocol Corpus — Vetted Sources & Licensing

Every chunk Wilderness Edge speaks aloud must cite an accredited source, and every source
must be one we are actually permitted to redistribute inside a shipped app binary. Those
are two different tests, and several well-known wilderness medicine texts pass the first
but fail the second.

`build_vector_db.py` enforces this by refusing to index any PDF that lacks an entry in
`sources.manifest.json`.

## Tier 1 — Public domain, unambiguous

Works of the US federal government are not subject to domestic copyright (17 U.S.C. § 105).
These can be bundled and redistributed freely, including commercially.

| Source | Publisher | Notes |
| --- | --- | --- |
| [ATP 4-02.11 — Casualty Response, TCCC, and First Aid](https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN46159-ATP_4-02.11-000-WEB-1.pdf) | HQ, Dept. of the Army | Published 23 Mar 2026; supersedes TC 4-02.1. Distribution unlimited. The best single trauma/first-aid backbone for this corpus. |
| [TCCC Handbook v5 (CALL 17-13)](https://api.army.mil/e2/c/downloads/2023/01/19/31e03488/17-13-tactical-casualty-combat-care-handbook-v5-may-17-distro-a.pdf) | US Army CALL | MARCH/PAWS algorithms, hemorrhage control, hypothermia prevention. Distribution A. |
| [TC 4-02.1 First Aid](https://armypubs.army.mil/epubs/DR_pubs/DR_a/ARN14135-TC_4-02.1-002-WEB-3.pdf) | HQ, Dept. of the Army | Superseded by ATP 4-02.11, but still public domain. Include only if you want historical coverage. |
| [Special Operations Forces Medical Handbook](https://archive.org/details/SOFMH2001) | USSOCOM | Marked Public Domain Mark 1.0 on Internet Archive. Large, OCR'd scan — extraction quality is poorer than born-digital PDFs; dry-run it before committing. |

## Tier 2 — Explicit publisher permission to harvest and adapt

| Source | Publisher | Permission |
| --- | --- | --- |
| [National Model EMS Clinical Guidelines v3.0](https://nasemso.org/wp-content/uploads/National-Model-EMS-Clinical-Guidelines_2022.pdf) | NASEMSO | The document itself states leaders are "invited to harvest content as will be useful" and may "adopt it in whole or in part." This is the closest legitimate stand-in for the "State EMS Protocols" named in the project mission. |
| Individual state EMS protocols (Maine, Utah, Montana, Virginia, Alaska, and others) | State EMS offices | Published openly, but **state** government works are not automatically public domain the way federal ones are. Licensing varies state by state — check each before adding, and record what you found in the manifest `license` field. |

## Tier 3 — Free to read, NOT free to bundle without asking first

| Source | Blocker |
| --- | --- |
| Hesperian, *Where There Is No Doctor* | Their Open Copyright Policy permits non-commercial copying and adaptation with credit, but explicitly requires prior permission to use any part "in any digital format." Bundling it in an app is exactly that case. Contact permissions@hesperian.org — for a non-commercial responder tool this is plausibly grantable, but it must be granted before you ship. Hesperian also asks that you link to their download page rather than rehost PDFs. |
| WMS Clinical Practice Guidelines (frostbite, pit viper envenomation, acute pain in austere environments, etc.) | Published in *Wilderness & Environmental Medicine* via SAGE. Individually excellent and topically perfect, but licensing is per-article — some are open access, most are not. Check the specific article's license before including any. |
| AHA / ILCOR CPR and ECC guidelines | Open to read in *Circulation*, but AHA retains copyright and actively licenses derivative use. |

## Tier 4 — Cannot be used as-is

**NOLS Wilderness First Aid / Wilderness First Responder curriculum.** `AGENTS.md` names NOLS
in the mission statement and `PLAN.md` uses `[Source: NOLS WFR Section 4.2]` as its running
citation example, but NOLS course materials and field texts are copyrighted and are not
released under any license permitting ingestion or redistribution. Using them would require a
direct license agreement with NOLS.

This matters beyond legality: the app's core safety promise is that it recites accredited
protocol text verbatim with a citation. Citing a manual you have not licensed is the one
failure mode that combines a legal problem with a safety-claim problem.

**Recommended fix:** keep NOLS as an aspirational partner-licensing target, and change the
documented example citation to a source actually in the corpus, e.g.
`[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]`.

## Adding a new source

1. Confirm the license permits redistribution in a compiled app. Record the exact permission
   language, not just "it's free online."
2. Add an entry to `sources.manifest.json` with `filename`, `title`, `publisher`, `license`,
   `url`, and `citation_prefix`.
3. Place the PDF in `sources/` under the manifest filename (or add a `url` and run
   `fetch_sources.py`).
4. Run `build_vector_db.py --dry-run` first and read the sampled chunks. Scanned PDFs without
   a text layer produce empty or garbled output, which is worse than omitting the source.

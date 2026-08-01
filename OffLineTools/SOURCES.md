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
| [FM 3-05.70 (FM 21-76) Survival](https://upload.wikimedia.org/wikipedia/commons/7/70/FM_3-05.70_%28FM_21-76%29_Survival_-_May_2002.pdf) | HQ, Dept. of the Army | **In the manifest, but read the caution below before ingesting.** Appendices C–F (poisonous plants, dangerous insects and arachnids, venomous snakes and lizards, dangerous fish) are the best public-domain flora/fauna hazard reference available and are already distilled into `species.manifest.json`. Fetched from the Wikimedia mirror because armypubs redirects; archive.org mirror recorded as a fallback. |
| [NWCG Incident Response Pocket Guide (IRPG), PMS 461](https://www.nwcg.gov/publications/pms461) | National Wildfire Coordinating Group | Federal interagency work. Already written as field checklists, which fits this app's output format better than anything else in the corpus: medical incident reporting, evacuation decision aids, LCES, the Risk Management Process. Manifest fetches a Washington DNR mirror because nwcg.gov serves an HTML landing page, not a direct PDF. |
| [Bad Bug Book, 2nd Edition](https://www.fda.gov/media/83271/download) | US Food and Drug Administration | Covers the ingestion hazards no first-aid manual in this corpus addresses: mushroom toxins, ciguatera, scombroid, shellfish poisoning. |

### Caution on FM 3-05.70 — read before ingesting it wholesale

Appendix B, *Edible and Medicinal Plants*, contains the **Universal Edibility Test**, which
modern wilderness medicine regards as unreliable and unsafe, plus medicinal-plant material
that is prescriptive in exactly the way this app must never be. A manual being public domain
does not make every page of it safe for an assistant to read aloud.

Three options, in descending order of preference:

1. **Don't ingest the PDF.** The valuable hazard content is already curated into
   `species.manifest.json` with citations back to this manual, without the foraging material.
   This is what the sprint build does by default.
2. **Ingest it and rely on the guard.** `SafetyFilter`'s `edibility_advice` pattern
   intercepts "safe to eat", "is edible", "you can eat", and "universal edibility test"
   before anything reaches TTS. Run `build_vector_db.py --dry-run` and actually read the
   sampled Appendix B chunks first.
3. **Ingest it with page exclusions.** `build_vector_db.py` has no page-range exclusion
   today; adding one is the clean long-term fix if the corpus takes on more mixed-safety
   sources.

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

## Flora & fauna hazard cards (`species.manifest.json`)

The protocol corpus tells a responder what to *do*. It says almost nothing about what they
are looking *at* — and "is this snake venomous", "is this plant the one that burned me" is
exactly what a camera-equipped assistant gets asked in the field. `species.manifest.json`
fills that gap with 20 hazard cards (urushiol plants, phototoxic and ingestion-toxic plants,
a mushroom, five snakes, two spiders, a tick, and four large mammals).

**Card text is original prose**, written here from the public-domain federal sources named
in each entry's `source_citation` (CDC/NIOSH Fast Facts, CDC tick removal guidance, US Army
FM 3-05.70 Appendices C–F, NPS wildlife safety guidance, FDA Bad Bug Book). Nothing is
copied from a commercial field guide — Audubon, Peterson, Sibley and the rest are all
copyrighted and none of them are licensable for this.

### Images

`build_species_pack.py` resolves reference photographs from Wikimedia Commons categories
and **hard-filters on the license the Commons API reports**. Only these are admitted:

| Admitted | Rejected |
| --- | --- |
| Public domain, CC0, CC BY (any version), CC BY-SA (any version) | Anything NonCommercial (`CC BY-NC*`), anything NoDerivatives (`*-ND`), fair-use claims, and any file whose license string does not parse |

NC and ND are excluded deliberately: this is a compiled binary that gets redistributed, and
those terms are incompatible with shipping it. The filter **fails closed** — an unrecognised
license string is rejected, and a species that ends up with zero admissible images fails the
build rather than shipping without imagery.

The current pinned set is 59 images across 20 species:

| License | Images |
| --- | --- |
| CC BY-SA 4.0 | 18 |
| Public domain | 9 |
| CC BY 4.0 | 8 |
| CC BY 2.0 | 6 |
| CC0 | 6 |
| CC BY-SA 3.0 | 5 |
| CC BY 3.0 | 4 |
| CC BY 2.5 | 2 |
| CC BY-SA 2.0 | 1 |

Resolved files are pinned in `species.images.lock.json` with their license and author, so
the set is reproducible and a license change shows up in a diff instead of silently
entering a build. Licenses are **re-verified against the lockfile on every build**, not
merely trusted from resolution time.

Two obligations follow from shipping CC BY / CC BY-SA images and are already implemented:

1. **Attribution must be visible wherever the image is.** `SpeciesCardView` renders
   `attribution — license` under every photograph. Do not remove it.
2. **Share-alike applies to the images, not to the app.** CC BY-SA covers the photographs
   as distributed; it does not reach the surrounding Swift code. Keep the per-image credit
   and the `source_url` intact and this stays clean.

Other federal image sources worth pulling from if the pack grows: the USFWS National Digital
Library, USGS (its Bee Inventory macro photography is public domain and exceptional), the
NPS image gallery, CDC PHIL, and Smithsonian Open Access (2.8M CC0 assets, though it needs
an api.data.gov key, which is why the sprint build uses Commons). iNaturalist and GBIF are
the best programmatic option at scale because the per-observation license is a queryable
field — but note that a large share of iNaturalist images are CC BY-NC and therefore fail
the filter above.

### The identification-safety problem

This is the part that matters more than the licensing. A small multimodal model asked
"what is this?" from one camera frame will answer confidently and will sometimes be wrong,
and "that one's non-venomous" is a failure that injures somebody. Species identification is
the same class of risk as diagnosis and is treated the same way:

- Card text describes **field marks and look-alikes** and ends by telling the responder that
  the assistant does not identify species.
- Cards for confusable pairs (black bear vs. grizzly, cottonmouth vs. water snake, coral
  snake vs. scarlet kingsnake) name the look-alike explicitly, because for bears the
  *response itself* differs and a misidentification changes what you tell someone to do.
- `SafetyFilter` gained three patterns — `species_id`, `harmless_reassurance`, and
  `edibility_advice` — that intercept the model asserting an identity, declaring something
  harmless, or giving foraging advice. Conditional framing ("if this is a copperhead, the
  protocol says…") is deliberately still allowed.
- `SpeciesCardView` shows the reference photographs as a "confirm this yourself" prompt.

## Adding a new source

1. Confirm the license permits redistribution in a compiled app. Record the exact permission
   language, not just "it's free online."
2. Add an entry to `sources.manifest.json` with `filename`, `title`, `publisher`, `license`,
   `url`, and `citation_prefix`.
3. Place the PDF in `sources/` under the manifest filename (or add a `url` and run
   `fetch_sources.py`).
4. Run `build_vector_db.py --dry-run` first and read the sampled chunks. Scanned PDFs without
   a text layer produce empty or garbled output, which is worse than omitting the source.

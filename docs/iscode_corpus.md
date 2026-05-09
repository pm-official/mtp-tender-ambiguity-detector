# IS code corpus assembly (Session 9 placeholder)

This file will document the IS-code corpus during Session 9 of the build.

For now: a stub. The corpus is built later; see the project methodology
document, Appendix D, for the version-pinned starter list.

## Layout

```
resources/iscode/
├── IS_456_2000.pdf            # Plain & Reinforced Concrete
├── IS_456_2000.metadata.json  # {code, version_year, sections, source_url}
├── IS_1786_2008.pdf
├── IS_1786_2008.metadata.json
├── ...
└── _index.json                # Top-level corpus index
```

## Version-pinning policy

(carried forward from the methodology document, Appendix D)

1. If the tender cites a specific version, the rewrite must cite that version.
2. If the tender does not cite a version, the rewrite cites the latest
   BIS-current version and surfaces this in the rewrite's `explanation` field.
3. Withdrawn / superseded codes are flagged at retrieval time and never used
   as the basis of a rewrite without an explicit warning.

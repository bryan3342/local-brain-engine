# Brain, implemented architecture

What was actually built, and why each part is shaped the way it is.

## The one idea

**Markdown is canonical. Everything else is derived and disposable.**

```
Obsidian vault (.md + YAML frontmatter)     <- source of truth
        |
        |  parsed, never authoritative
        v
SQLite + FTS5  ·  backlinks  ·  graph export  ·  (future embeddings)
```

There is a test that proves this rather than asserting it: delete the database,
rebuild, and `list`, `backlinks`, `export-graph` and `search` return byte-identical
results. Another runs the entire API with `--no-index`.

## Layers

```
brain/
  config.py            BrainConfig, injected, never a module singleton
  cli.py               argparse; ALL printing in the package lives here

  domain/              pure, no I/O
    enums.py           NoteType · NoteStatus · RelationshipType
    errors.py          typed BrainError hierarchy -> CLI exit codes, later MCP errors
    models.py          Note · Frontmatter · Relationship · reports (all .to_dict())
    identity.py        slugs, id minting, edge encoding, anchors   [highest-risk pure code]

  application/
    knowledge_api.py   the facade, the only writer that maintains invariants
    mapper.py          YAML mapping <-> domain models
    validators.py      metadata validation + duplicate detection
    reconcile.py       scan() + doctor(), two-writer convergence

  infrastructure/      the only code that touches disk / sqlite / git
    markdown.py        head-anchored split + ruamel round-trip   [load-bearing contract]
    vault.py           atomic write, CAS, flock, git
    index.py           derived SQLite + FTS5
    templates.py       create-only bodies
    audit_log.py       append-only JSONL
```

One runtime dependency: `ruamel.yaml`. Everything else is stdlib.

## The invariants, and what enforces each

| # | Invariant | Enforced by | Test |
|---|---|---|---|
| C1 | `id` minted once, frozen forever; never re-derived from a title | `identity.mint_id` is create-only; `update_note`/`rename_note` re-pin `id` after every splice | rename keeps id, backlinks, and aliases the old title |
| C2 | Three writers exist; the API converges rather than pretends | tolerant `iter_notes`, `Reconciler.scan/doctor` | hand-written file is adopted, body untouched |
| C3 | Canonical store survives a crash and a concurrent writer | temp → `fsync` → `os.replace`, dir fsync, `(mtime_ns, size, sha256)` CAS, `flock` | `os.replace` monkeypatched to raise; original intact, no temp leaked |
| H1a | `---` inside a code fence cannot truncate a body | head-anchored split; body is an opaque byte range | fenced-`---` corpus + hypothesis over arbitrary bytes |
| H1b | Update never re-renders a body | `splice()` mutates the mapping and re-emits body bytes | body byte-identical after a frontmatter-only update |
| H1c/e | Writes do not churn the file; reads tolerate what Obsidian does | ruamel round-trip, `width` raised so long scalars never reflow; `coerce_date` accepts quoted, bare, and already-coerced dates | key order and quotes preserved; 400-char scalar stays on one line |
| H1d | Edges are flat `type::target` strings, never nested YAML | `Relationship.encode()` |, |
| H2 | Rollback exists | `git init` at vault creation; the audit log deliberately stores no pre-image | git commit round-trip |
| H3 | Reads are not O(vault) | derived SQLite+FTS5; backlinks are O(indegree); link does not validate targets | `rm brain.db` equivalence; touch does not reparse |
| H4 | `schema_version` and structured `source` ship from note #1 | `mapper.CANONICAL_ORDER` |, |
| M1 | Type is authoritative; folder is cosmetic | `update_note` never moves a file; folder mismatch is advisory and off by default | advisory hidden unless asked |
| M2 | Duplicates are surfaced, never auto-merged | `(type, normalized_title)` + alias cross-check | dedupe scoped by type |
| M3 | Soft-delete is defined | `status: deleted` + move to trash; inbound edges become `unresolved` | deleted note is not a ghost node |

## Anchors: why the graph has more nodes than notes

An edge may point at something that is not a note, a file a session touched, a
PR it shipped, a skill it used, the domain it belongs to. These are emitted as
synthetic nodes (`anchor:file`, `anchor:pr`, …) rather than reported as broken.

That decision is what makes multi-hop retrieval work, because two sessions that
touched the same file are only connected *through* that file. On the real vault:

| | notes only | with anchors |
|---|---|---|
| nodes | 263 | **754** |
| edges | 45 | **1,154** |
| unresolved | 1,118 | **9** |

## Resolution order

`id` → `alias` → `path` → normalized title. It deliberately never includes
"re-slugify the title": that is exactly the C1 bug, where renaming a note in
Obsidian silently forks the graph.

Edges resolve the same way. That fixed a real defect found by running `scan`
against the real vault, bridged memory notes referenced their origin session by
uuid while session reports are keyed by date and slug, so 78 provenance edges
dangled.

## Next seam: MCP, not HTTP

`KnowledgeApi` is a pure library: explicit config, no printing, no global state,
JSON-serializable returns, typed errors. An MCP server exposing `create_note` /
`search_notes` / `link_notes` / `get_backlinks` as tools is a thin adapter over
it. HTTP would mean inventing transport, auth and schemas and then wrapping them
in MCP anyway.

"""Thin CLI. Every bit of formatting and printing in the package lives here.

The API layer stays a pure library that returns dataclasses, so the same
operations can be exposed over MCP without refactoring anything (the Phase-2
seam is MCP, not HTTP, Claude speaks MCP natively, and an HTTP API would mean
inventing transport, auth and schemas only to wrap them in MCP afterwards).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from brain.application.knowledge_api import KnowledgeApi
from brain.config import BrainConfig
from brain.domain.errors import BrainError


def _Sub(parent: argparse.ArgumentParser) -> type[argparse.ArgumentParser]:
    """Subparser class that inherits the global flags."""

    class _WithGlobals(argparse.ArgumentParser):
        def __init__(self, **kwargs: Any) -> None:
            kwargs.setdefault("parents", [parent])
            super().__init__(**kwargs)

    return _WithGlobals


EXIT_CODES: dict[str, int] = {
    "NoteNotFoundError": 3,
    "DuplicateNoteError": 4,
    "ValidationError": 5,
    "ConcurrentModificationError": 6,
    "FrontmatterError": 7,
    "ImmutableFieldError": 8,
    "LockError": 9,
    "VaultError": 10,
    "CandidateNotFoundError": 11,
}


def _api(args: argparse.Namespace) -> KnowledgeApi:
    from brain.infrastructure import embedder as embedder_module
    from brain.infrastructure.index import SqliteIndex

    config = BrainConfig.load(args.vault)
    # Resolved through the module rather than imported by name so the choice
    # stays observable, and overridable, at the seam it is made.
    index = (None if args.no_index
             else SqliteIndex(config.root, embedder=embedder_module.embedder_from_env()))
    return KnowledgeApi(config, index=index)


def _emit(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
        return
    if isinstance(value, list):
        for item in value:
            print(_line(item))
    elif isinstance(value, dict):
        for key, val in value.items():
            print(f"{key}: {val}" if not isinstance(val, list) else f"{key}: {len(val)}")
    else:
        print(value)


def _line(item: Any) -> str:
    if isinstance(item, dict):
        if "title" in item and "id" in item:
            answer = item.get("answer") or item.get("snippet") or ""
            answer = (answer[:100] + "…") if len(answer) > 100 else answer
            return (
                f"{item['id']:<52} {item['title']}"
                + (f"\n{'':<52} {answer}" if answer else "")
            )
        return json.dumps(item, ensure_ascii=False, default=str)
    return str(item)


def _print_stale(stale: list[str]) -> None:
    """The warning that a golden case expects a note that no longer exists.

    Shared by `eval` and `eval --coverage` so the two modes cannot drift apart:
    a stale expectation is a defect in the golden set, not in retrieval, and
    coverage mode is exactly when someone is auditing the golden set and most
    needs to be told.
    """
    if not stale:
        return
    print(
        f"⚠ {len(stale)} expected note(s) do not exist, "
        "fix the golden set, not the vault:"
    )
    for note_id in stale:
        print(f"    {note_id}")
    print()


def _print_eval(report: Any, stale: list[str], misses_only: bool) -> None:
    """Render an eval report. Misses first, they are the actionable part."""
    _print_stale(stale)
    if report.misses:
        print(f"MISSES ({len(report.misses)} of {report.total})")
        for r in report.misses:
            print(f"  ✗ {r.case.query}")
            print(f"      expected: {', '.join(r.case.expect)}")
            print(f"      returned: {', '.join(r.returned) or '(nothing)'}")
        print()
    if not misses_only:
        low = [r for r in report.results if r.rank and r.rank > 1]
        if low:
            print(f"RANKED BELOW 1 ({len(low)})")
            for r in sorted(low, key=lambda x: -(x.rank or 0)):
                print(f"  ~ rank {r.rank}  {r.case.query}")
            print()
    print(f"cases        {report.total}")
    print(f"precision@1  {report.precision_at_1:.1%}")
    print(f"recall@{report.k}     {report.recall_at_k:.1%}")
    print(f"MRR          {report.mrr:.3f}")


def build_parser() -> argparse.ArgumentParser:
    # Global flags live on a parent parser so they are accepted either before or
    # after the subcommand. `brain reindex --json` is what people actually type.
    # SUPPRESS is load-bearing: the subparser inherits these actions, and with a
    # normal default it would reset a flag the top-level parser already set,
    # silently turning `brain --json list` back off. With SUPPRESS the attribute
    # is written only where the user actually passed it; _defaults() fills the rest.
    globals_ = argparse.ArgumentParser(add_help=False)
    globals_.add_argument(
        "--vault", default=argparse.SUPPRESS,
        help="vault root (default: $BRAIN_VAULT)"
    )
    globals_.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS,
        help="emit JSON"
    )
    globals_.add_argument(
        "--no-index", action="store_true", default=argparse.SUPPRESS,
        help="bypass the derived index (proves it is disposable)"
    )

    p = argparse.ArgumentParser(
        prog="brain", parents=[globals_],
        description="Local-first personal knowledge OS."
    )
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Sub(globals_))

    sub.add_parser("init", help="create the vault skeleton")

    c = sub.add_parser("create", help="create a note")
    c.add_argument("title")
    c.add_argument("--type", default="concept")
    c.add_argument("--domain")
    c.add_argument("--status", default="draft")
    c.add_argument("--tag", action="append", default=[])
    c.add_argument("--alias", action="append", default=[])
    c.add_argument("--rel", action="append", default=[], metavar="REL::TARGET")
    c.add_argument("--answer", default="")
    c.add_argument("--folder")
    c.add_argument("--allow-duplicate", action="store_true")

    g = sub.add_parser("get", help="show one note")
    g.add_argument("ref")
    g.add_argument("--body", action="store_true")

    u = sub.add_parser(
        "update", help="update frontmatter (body is never touched)"
    )
    u.add_argument("ref")
    u.add_argument("--title")
    u.add_argument("--status")
    u.add_argument("--domain")
    u.add_argument("--tag", action="append")
    u.add_argument("--alias", action="append")

    r = sub.add_parser("rename", help="retitle and/or move; id is preserved")
    r.add_argument("ref")
    r.add_argument("new_title")
    r.add_argument("--folder")

    d = sub.add_parser("delete", help="soft-delete to trash")
    d.add_argument("ref")
    d.add_argument("--hard", action="store_true")

    ls = sub.add_parser("list", help="list notes")
    ls.add_argument("--type")
    ls.add_argument("--domain")
    ls.add_argument("--status")
    ls.add_argument("--limit", type=int)

    s = sub.add_parser("search", help="full-text search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=20)

    ln = sub.add_parser("link", help="add a typed edge")
    ln.add_argument("src")
    ln.add_argument("rel")
    ln.add_argument("dst")

    ul = sub.add_parser("unlink", help="remove a typed edge")
    ul.add_argument("src")
    ul.add_argument("rel")
    ul.add_argument("dst")

    b = sub.add_parser("backlinks", help="inbound edges")
    b.add_argument("ref")

    n = sub.add_parser("neighbors", help="inbound and outbound edges")
    n.add_argument("ref")

    sub.add_parser("export-graph", help="emit nodes and edges as JSON")

    ri = sub.add_parser("reindex", help="rebuild the derived index")
    ri.add_argument("--full", action="store_true")

    sc = sub.add_parser("scan", help="report files the API did not write")
    sc.add_argument(
        "--advisory", action="store_true",
        help="include advisory findings such as type/folder mismatch"
    )

    doc = sub.add_parser("doctor", help="propose (or apply) repairs")
    doc.add_argument(
        "--apply", action="store_true", help="actually write the repairs"
    )

    ev = sub.add_parser("eval", help="score retrieval against the golden query set")
    ev.add_argument(
        "--golden", default=None,
        help="path to golden.jsonl (default: <vault>/90-meta/eval/golden.jsonl)"
    )
    ev.add_argument("-k", type=int, default=5, help="how deep a hit still counts")
    ev.add_argument(
        "--misses-only", action="store_true", help="print only the failures"
    )
    ev.add_argument(
        "--coverage", action="store_true",
        help="list notes no golden query retrieves"
    )
    ev.add_argument(
        "--save-baseline", action="store_true",
        help="record the current score as the bar to beat"
    )
    ev.add_argument(
        "--check", action="store_true",
        help="compare against the saved baseline; non-zero if it regressed"
    )

    es = sub.add_parser("etl-scan", help="read new sessions into the ETL queue")
    es.add_argument("--limit", type=int, default=500)
    es.add_argument(
        "--db", default=None, help="episodic-memory db (default: standard path)"
    )

    sub.add_parser("etl-status", help="how much work is queued")

    # No --folder or --status here, ever: placement is apply()'s decision, not
    # the skill's. That is the entire point of this subcommand existing
    # instead of the skill calling `create` with model-authored argv.
    ed = sub.add_parser(
        "etl-draft",
        help="write a reconciled note from a queued candidate"
    )
    ed.add_argument("--candidate-id", required=True)
    ed.add_argument("--title", required=True)
    ed.add_argument("--type", default="concept")
    ed.add_argument("--domain", default="")
    ed.add_argument("--answer", default="")
    ed.add_argument("--alias", action="append", default=[])
    ed.add_argument("--tag", action="append", default=[])
    ed.add_argument("--rel", action="append", default=[], metavar="REL::TARGET")
    # Placement is still not the caller's to choose -- this names WHICH note the
    # new one contradicts, not where either of them lives. The archive happens
    # at promotion, when a human agrees.
    ed.add_argument(
        "--supersedes", default=None, metavar="NOTE_ID",
        help="id of a note this one contradicts and should replace"
    )

    rv = sub.add_parser(
        "etl-revert",
        help="undo the aliases one ETL session added"
    )
    rv.add_argument(
        "--session", required=True,
        help="source session id, as recorded in the audit log"
    )

    sub.add_parser("inbox", help="list candidates awaiting promotion")

    pr = sub.add_parser("promote", help="promote a candidate out of the inbox")
    pr.add_argument("ref")
    pr.add_argument("--to", help="destination folder (default: by type)")

    rj = sub.add_parser("reject", help="reject a candidate, with a reason")
    rj.add_argument("ref")
    rj.add_argument("reason")

    th = sub.add_parser("teach", help="record that a query should find a note")
    th.add_argument("query")
    th.add_argument("note_id")

    sub.add_parser("audit", help="tail the audit log")
    return p


def _defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill globals the user did not pass (see the SUPPRESS note in build_parser)."""
    for name, default in (("vault", None), ("json", False), ("no_index", False)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _defaults(build_parser().parse_args(argv))
    try:
        api = _api(args)
        cmd = args.command

        if cmd == "init":
            _emit(api.initialize_vault(), args.json)
        elif cmd == "create":
            note = api.create_note(
                args.title, args.type, domain=args.domain, status=args.status,
                tags=args.tag, aliases=args.alias, relationships=args.rel,
                answer=args.answer, folder=args.folder,
                allow_duplicate=args.allow_duplicate)
            _emit(note.to_dict(), args.json)
        elif cmd == "get":
            _emit(api.get_note(args.ref).to_dict(include_body=args.body), args.json)
        elif cmd == "update":
            note = api.update_note(
                args.ref, title=args.title, status=args.status,
                domain=args.domain, tags=args.tag, aliases=args.alias)
            _emit(note.to_dict(), args.json)
        elif cmd == "rename":
            _emit(api.rename_note(
                args.ref, args.new_title,
                dest_folder=args.folder).to_dict(), args.json)
        elif cmd == "delete":
            _emit(api.delete_note(args.ref, hard=args.hard), args.json)
        elif cmd == "list":
            _emit([r.to_dict() for r in api.list_notes(
                note_type=args.type, domain=args.domain,
                status=args.status, limit=args.limit)], args.json)
        elif cmd == "search":
            _emit(
                [h.to_dict() for h in api.search_notes(args.query, limit=args.limit)],
                args.json
            )
        elif cmd == "link":
            _emit(
                api.link_notes(args.src, args.rel, args.dst).to_dict(), args.json
            )
        elif cmd == "unlink":
            _emit(
                api.unlink_notes(args.src, args.rel, args.dst).to_dict(), args.json
            )
        elif cmd == "backlinks":
            _emit([b.to_dict() for b in api.get_backlinks(args.ref)], args.json)
        elif cmd == "neighbors":
            _emit(api.get_neighbors(args.ref), True)
        elif cmd == "export-graph":
            _emit(api.export_graph().to_dict(), True)
        elif cmd == "reindex":
            _emit(api.reindex(full=args.full).to_dict(), args.json)
        elif cmd == "scan":
            from brain.application.reconcile import Reconciler

            _emit(Reconciler(api).scan(include_advisory=args.advisory).to_dict(), args.json)
        elif cmd == "doctor":
            from brain.application.reconcile import Reconciler

            _emit(Reconciler(api).doctor(dry_run=not args.apply).to_dict(), args.json)
        elif cmd == "eval":
            from brain.application.evaluate import (
                GOLDEN_PATH,
                discover_eval_sets,
                evaluate,
                load_cases,
                unresolved_expectations,
            )

            golden = (
                Path(args.golden) if args.golden
                else api.vault.root / GOLDEN_PATH
            )
            cases = load_cases(golden)
            stale = list(unresolved_expectations(api, cases))
            if args.save_baseline:
                from brain.etl.learn import save_baseline

                _emit(save_baseline(api, cases), args.json)
                return 0
            if args.check:
                from brain.etl.learn import check_against_baseline

                verdict = check_against_baseline(api, cases)
                _print_stale(stale) if not args.json else None
                _emit(verdict, args.json)
                # Non-zero on regression so a hook or CI step can act on it
                # without parsing output.
                return 1 if verdict["regressed"] else 0
            if args.coverage:
                from brain.etl.learn import coverage

                unreached = coverage(api, cases, limit=args.k)
                if args.json:
                    _emit({"unreached": unreached, "stale_expectations": stale}, True)
                else:
                    _print_stale(stale)
                    _emit({"unreached": unreached}, False)
                return 0
            # Report every set the vault has, not just the one retrieval was
            # tuned against. `golden.jsonl` was written alongside the alias work
            # and knows the vocabulary already indexed, so it scores well for a
            # reason that is not retrieval quality. An explicit --golden still
            # gets the flat single-set shape a script would be parsing.
            sets = (
                [(golden.stem, golden)] if args.golden
                else discover_eval_sets(api.vault.root) or [(golden.stem, golden)]
            )
            scored = []
            for name, path in sets:
                set_cases = cases if path == golden else load_cases(path)
                set_stale = (
                    stale if path == golden
                    else list(unresolved_expectations(api, set_cases))
                )
                scored.append((name, evaluate(api, set_cases, k=args.k), set_stale))

            if args.json:
                if args.golden:
                    _, report, set_stale = scored[0]
                    _emit(
                        {**report.to_dict(), "stale_expectations": set_stale}, True
                    )
                else:
                    _emit({"sets": {name: {**report.to_dict(),
                                           "stale_expectations": set_stale}
                                    for name, report, set_stale in scored}}, True)
            else:
                for index, (name, report, set_stale) in enumerate(scored):
                    if len(scored) > 1:
                        if index:
                            print()
                        print(f"── {name} ──")
                    _print_eval(report, set_stale, args.misses_only)
            return 0 if all(not report.misses for _, report, _ in scored) else 1
        elif cmd == "etl-scan":
            from brain.etl.extract import scan
            from brain.etl.queue import Queue

            stats = scan(
                api, Queue(api.vault),
                db_path=Path(args.db) if args.db else None, limit=args.limit
            )
            _emit(stats, args.json)
        elif cmd == "etl-status":
            from brain.etl.queue import Queue

            q = Queue(api.vault)
            _emit({"pending": len(q.pending()),
                   "rejected_total": len(Queue.read_jsonl(q.rejected_path)),
                   "cursor": q.cursor()}, args.json)
        elif cmd == "etl-draft":
            from brain.etl.queue import Queue
            from brain.etl.reconcile import draft_from_candidate

            result = draft_from_candidate(
                api, Queue(api.vault), args.candidate_id,
                title=args.title, note_type=args.type, domain=args.domain,
                answer=args.answer, aliases=args.alias, tags=args.tag,
                relationships=args.rel, supersedes=args.supersedes)
            _emit(result, args.json)
        elif cmd == "etl-revert":
            from brain.etl.learn import revert_session

            _emit(revert_session(api, args.session), args.json)
        elif cmd == "inbox":
            from brain.etl.inbox import list_inbox

            _emit(list_inbox(api), args.json)
        elif cmd == "promote":
            from brain.etl.inbox import promote

            _emit(promote(api, args.ref, to=args.to), args.json)
        elif cmd == "reject":
            from brain.etl.inbox import reject
            from brain.etl.queue import Queue

            _emit(reject(api, Queue(api.vault), args.ref, args.reason), args.json)
        elif cmd == "teach":
            from brain.etl.learn import teach

            _emit(teach(api, args.query, args.note_id), args.json)
        elif cmd == "audit":
            _emit(api.audit.tail(30), args.json)
        else:  # pragma: no cover - argparse enforces this
            return 2
        return 0
    except BrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CODES.get(type(exc).__name__, 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

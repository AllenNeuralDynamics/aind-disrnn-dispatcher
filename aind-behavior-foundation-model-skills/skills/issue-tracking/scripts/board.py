#!/usr/bin/env python3
"""File an issue in an AIND dynamic-foraging-bfm repo and put it on the AIND-behavior-fm board.

Does the four steps in the issue-tracking skill: create issue -> add to project ->
set Status/Priority/Size -> read back and verify. Needs GITHUB_TOKEN with repo+project
scopes, SSO-authorized for AllenNeuralDynamics.

    python board.py --repo aind-dynamic-foraging-bfm-dispatcher --title "..." --body-file issue.md \
        --labels documentation priority:P1 --status "In progress" --priority P1 --size M

    python board.py --discover                 # re-read field/option ids from the board
    python board.py --existing 88 --status "In review"   # move an already-filed issue
    python board.py --existing 88 --check codebase-map --check wrapper-runtime
    python board.py --existing 88 --check-all --status Done --close
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ORG = "AllenNeuralDynamics"
PROJECT_NUMBER = 184
DEFAULT_ASSIGNEE = "hanhou"
# Cached 2026-09-01; --discover re-reads them.
PROJECT_ID = "PVT_kwDOBa47bs4BIeG5"
FIELDS = {
    "Status": ("PVTSSF_lADOBa47bs4BIeG5zg4672s", {
        "Backlog": "f75ad846", "Ready": "61e4505c", "In progress": "47fc9ee4",
        "In review": "df73e18b", "Done": "98236657"}),
    "Priority": ("PVTSSF_lADOBa47bs4BIeG5zg467-8", {
        "P0": "79628723", "P1": "0a877460", "P2": "da944a9c"}),
    "Size": ("PVTSSF_lADOBa47bs4BIeG5zg467_A", {
        "XS": "6c6483d2", "S": "f784b110", "M": "7515a9f1",
        "L": "817d0097", "XL": "db339eb2"}),
}
DISCOVER_QUERY = """
query($org:String!, $num:Int!){
  organization(login:$org){ projectV2(number:$num){
    id title
    fields(first:30){ nodes{
      ... on ProjectV2FieldCommon { id name dataType }
      ... on ProjectV2SingleSelectField { id name options{ id name } } }}
  }}}"""
ADD_ITEM = """
mutation($p:ID!, $c:ID!){
  addProjectV2ItemById(input:{projectId:$p, contentId:$c}){ item{ id } } }"""
SET_FIELD = """
mutation($p:ID!, $i:ID!, $f:ID!, $o:String!){
  updateProjectV2ItemFieldValue(input:{
    projectId:$p, itemId:$i, fieldId:$f, value:{singleSelectOptionId:$o}
  }){ projectV2Item{ id } } }"""
READ_ITEM = """
query($i:ID!){ node(id:$i){ ... on ProjectV2Item {
  content{ ... on Issue { number url } }
  fieldValues(first:20){ nodes{
    ... on ProjectV2ItemFieldSingleSelectValue {
      name field{ ... on ProjectV2SingleSelectField { name } } } }}
}}}"""


def _token():
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("GITHUB_TOKEN is not set.")
    return tok


def _request(url, payload=None, method="GET"):
    req = urllib.request.Request(
        url, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + _token(),
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        if exc.code == 403 and "SAML" in detail:
            sys.exit("403: the PAT is not SSO-authorized for " + ORG
                     + ". Authorize it in the token settings, then rerun.")
        sys.exit("HTTP {}: {}".format(exc.code, detail))


def gql(query, variables):
    """GraphQL call. Errors arrive with HTTP 200, so check the errors array."""
    out = _request("https://api.github.com/graphql",
                   {"query": query, "variables": variables}, method="POST")
    if out.get("errors"):
        sys.exit("GraphQL error: " + json.dumps(out["errors"])[:500])
    return out["data"]


def discover():
    proj = gql(DISCOVER_QUERY, {"org": ORG, "num": PROJECT_NUMBER})["organization"]["projectV2"]
    print("project {!r} id={}".format(proj["title"], proj["id"]))
    for field in proj["fields"]["nodes"]:
        if not field.get("name"):
            continue
        print("  {:<22} {}".format(field["name"], field["id"]))
        for opt in field.get("options") or []:
            print("      {:<14} {}".format(opt["name"], opt["id"]))
    return proj


def create_issue(repo, title, body, labels, assignees=None):
    if assignees is None:
        assignees = [DEFAULT_ASSIGNEE]
    issue = _request(
        "https://api.github.com/repos/{}/{}/issues".format(ORG, repo),
        {"title": title, "body": body, "labels": labels or [],
         "assignees": assignees}, method="POST")
    print("issue #{}: {}".format(issue["number"], issue["html_url"]))
    got = [a["login"] for a in issue.get("assignees") or []]
    if set(assignees) - set(got):
        # GitHub silently drops assignees the repo can't assign; say so rather than
        # leaving an unassigned issue that looks assigned.
        print("  WARNING: requested {} but GitHub set {} — assign by hand"
              .format(assignees, got or "nobody"))
    else:
        print("  assigned: {}".format(", ".join(got)))
    return issue


def ensure_assignee(repo, number, current, assignees=None):
    """Add the assignee to an existing issue if it has none of them."""
    if assignees is None:
        assignees = [DEFAULT_ASSIGNEE]
    have = [a["login"] for a in current or []]
    missing = [a for a in assignees if a not in have]
    if not missing:
        print("  assigned: {}".format(", ".join(have)))
        return
    out = _request(
        "https://api.github.com/repos/{}/{}/issues/{}/assignees".format(ORG, repo, number),
        {"assignees": missing}, method="POST")
    print("  assigned: {}".format(
        ", ".join(a["login"] for a in out.get("assignees") or [])))


def get_issue(repo, number):
    return _request("https://api.github.com/repos/{}/{}/issues/{}".format(ORG, repo, number))


def add_to_board(issue_node_id):
    return gql(ADD_ITEM, {"p": PROJECT_ID, "c": issue_node_id})[
        "addProjectV2ItemById"]["item"]["id"]


def set_fields(item_id, values):
    """values: {"Status": "In progress", "Priority": "P1", "Size": "M"} (None = skip)."""
    for name, choice in values.items():
        if choice is None:
            continue
        field_id, options = FIELDS[name]
        if choice not in options:
            sys.exit("{} must be one of {}".format(name, sorted(options)))
        gql(SET_FIELD, {"p": PROJECT_ID, "i": item_id,
                        "f": field_id, "o": options[choice]})


def tick_boxes(repo, number, body, substrings, check_all=False):
    """Flip '- [ ]' -> '- [x]' on unticked lines matching each substring.

    Only the matched lines change, so concurrent browser edits elsewhere in the body
    survive. A substring matching no unticked box is an error rather than a silent
    no-op -- a typo there would otherwise look like success.
    """
    lines = body.splitlines()
    unticked = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("- [ ]")]
    if not unticked:
        print("no unticked boxes in #{}".format(number))
        return body, []
    if check_all:
        targets = list(unticked)
    else:
        targets, missing = [], []
        for sub in substrings:
            hits = [i for i in unticked if sub.lower() in lines[i].lower()]
            if not hits:
                missing.append(sub)
            targets.extend(hits)
        if missing:
            sys.exit("no unticked box matches: {}\nunticked boxes are:\n{}".format(
                ", ".join(repr(m) for m in missing),
                "\n".join("  " + lines[i].strip() for i in unticked)))
    for i in sorted(set(targets)):
        lines[i] = lines[i].replace("- [ ]", "- [x]", 1)
        print("  ticked: {}".format(lines[i].strip()[:88]))
    return "\n".join(lines) + ("\n" if body.endswith("\n") else ""), sorted(set(targets))


def update_body(repo, number, body):
    _request("https://api.github.com/repos/{}/{}/issues/{}".format(ORG, repo, number),
             {"body": body}, method="PATCH")


def close_issue(repo, number):
    _request("https://api.github.com/repos/{}/{}/issues/{}".format(ORG, repo, number),
             {"state": "closed"}, method="PATCH")
    print("closed #{}".format(number))


def remaining_boxes(body):
    return [ln.strip() for ln in body.splitlines() if ln.lstrip().startswith("- [ ]")]


def verify(item_id, expected=None):
    """Read the item back and CONFIRM the requested values landed.

    A field mutation returns success for a valid-but-wrong option id, so printing
    the result is not verification -- the values have to be compared. Exits non-zero
    on mismatch so a stale cached option id fails loudly instead of silently
    setting the wrong field.
    """
    node = gql(READ_ITEM, {"i": item_id})["node"]
    got = {fv["field"]["name"]: fv["name"]
           for fv in node["fieldValues"]["nodes"] if fv.get("field")}
    print("on board: #{} {}".format(node["content"]["number"], got))
    wrong = []
    for name, want in (expected or {}).items():
        if want is None:
            continue
        if got.get(name) != want:
            wrong.append("{}: requested {!r}, board has {!r}".format(
                name, want, got.get(name)))
    if wrong:
        sys.exit("board field mismatch — the cached option IDs may be stale, "
                 "re-read them with --discover:\n  " + "\n  ".join(wrong))
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true",
                    help="print the board's field and option ids, then exit")
    ap.add_argument("--repo", default="aind-dynamic-foraging-bfm-dispatcher")
    ap.add_argument("--existing", type=int,
                    help="use this existing issue number instead of creating one")
    ap.add_argument("--title")
    ap.add_argument("--body-file", help="path to a markdown file for the issue body")
    ap.add_argument("--body", default="")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--assignee", action="append", default=None, metavar="LOGIN",
                    help="issue assignee; repeatable. Defaults to "
                         + DEFAULT_ASSIGNEE + " — every issue gets an owner.")
    ap.add_argument("--status", choices=sorted(FIELDS["Status"][1]))
    ap.add_argument("--priority", choices=sorted(FIELDS["Priority"][1]))
    ap.add_argument("--size", choices=sorted(FIELDS["Size"][1]))
    ap.add_argument("--check", action="append", default=[], metavar="SUBSTRING",
                    help="tick the unticked 'Done when' box(es) matching this substring; "
                         "repeatable. Requires --existing.")
    ap.add_argument("--check-all", action="store_true",
                    help="tick every remaining box (use when the last work has landed)")
    ap.add_argument("--close", action="store_true",
                    help="close the issue; refuses while boxes remain unticked")
    args = ap.parse_args()

    if args.discover:
        discover()
        return

    if (args.check or args.check_all or args.close) and not args.existing:
        sys.exit("--check/--check-all/--close need --existing <issue number>")

    if args.existing:
        issue = get_issue(args.repo, args.existing)
        print("issue #{}: {}".format(issue["number"], issue["html_url"]))
        ensure_assignee(args.repo, args.existing, issue.get("assignees"), args.assignee)
        body = issue.get("body") or ""
        if args.check or args.check_all:
            body, ticked = tick_boxes(args.repo, args.existing, body,
                                      args.check, check_all=args.check_all)
            if ticked:
                update_body(args.repo, args.existing, body)
        left = remaining_boxes(body)
        if left:
            print("{} box(es) still open:".format(len(left)))
            for ln in left:
                print("  " + ln[:88])
        if args.close:
            if left:
                sys.exit("refusing to close #{}: {} box(es) still unticked. Tick them, "
                         "strike them with a reason, or move them to a follow-up issue."
                         .format(args.existing, len(left)))
            close_issue(args.repo, args.existing)
    else:
        if not args.title:
            sys.exit("--title is required unless --existing is given")
        if args.body_file:
            with open(args.body_file, encoding="utf-8") as fh:
                body = fh.read()
        else:
            body = args.body
        issue = create_issue(args.repo, args.title, body, args.labels, args.assignee)

    requested = {"Status": args.status, "Priority": args.priority, "Size": args.size}
    item_id = add_to_board(issue["node_id"])
    set_fields(item_id, requested)
    verify(item_id, requested)


if __name__ == "__main__":
    main()

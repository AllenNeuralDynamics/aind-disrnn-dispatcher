#!/usr/bin/env python3
"""File an issue in an AIND disRNN repo and put it on the AIND-behavior-fm board.

Does the four steps in the issue-tracking skill: create issue -> add to project ->
set Status/Priority/Size -> read back and verify. Needs GITHUB_TOKEN with repo+project
scopes, SSO-authorized for AllenNeuralDynamics.

    python board.py --repo aind-disrnn-dispatcher --title "..." --body-file issue.md \
        --labels documentation priority:P1 --status "In progress" --priority P1 --size M

    python board.py --discover                 # re-read field/option ids from the board
    python board.py --existing 88 --status "In review"   # move an already-filed issue
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

ORG = "AllenNeuralDynamics"
PROJECT_NUMBER = 184
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


def create_issue(repo, title, body, labels):
    issue = _request(
        "https://api.github.com/repos/{}/{}/issues".format(ORG, repo),
        {"title": title, "body": body, "labels": labels or []}, method="POST")
    print("issue #{}: {}".format(issue["number"], issue["html_url"]))
    return issue


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


def verify(item_id):
    node = gql(READ_ITEM, {"i": item_id})["node"]
    got = {fv["field"]["name"]: fv["name"]
           for fv in node["fieldValues"]["nodes"] if fv.get("field")}
    print("on board: #{} {}".format(node["content"]["number"], got))
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true",
                    help="print the board's field and option ids, then exit")
    ap.add_argument("--repo", default="aind-disrnn-dispatcher")
    ap.add_argument("--existing", type=int,
                    help="use this existing issue number instead of creating one")
    ap.add_argument("--title")
    ap.add_argument("--body-file", help="path to a markdown file for the issue body")
    ap.add_argument("--body", default="")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--status", choices=sorted(FIELDS["Status"][1]))
    ap.add_argument("--priority", choices=sorted(FIELDS["Priority"][1]))
    ap.add_argument("--size", choices=sorted(FIELDS["Size"][1]))
    args = ap.parse_args()

    if args.discover:
        discover()
        return

    if args.existing:
        issue = get_issue(args.repo, args.existing)
        print("issue #{}: {}".format(issue["number"], issue["html_url"]))
    else:
        if not args.title:
            sys.exit("--title is required unless --existing is given")
        body = open(args.body_file).read() if args.body_file else args.body
        issue = create_issue(args.repo, args.title, body, args.labels)

    item_id = add_to_board(issue["node_id"])
    set_fields(item_id, {"Status": args.status,
                         "Priority": args.priority, "Size": args.size})
    verify(item_id)


if __name__ == "__main__":
    main()

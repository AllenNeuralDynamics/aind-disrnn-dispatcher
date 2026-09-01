# ProjectV2 board API — raw calls

Endpoint `https://api.github.com/graphql`, header `Authorization: Bearer $GITHUB_TOKEN`.
REST is `https://api.github.com`. No `gh` CLI in the sandbox.

## Discover fields and option ids

Run this after anyone edits the board's fields; it is the source of the cached table in
`SKILL.md`.

```graphql
query($org:String!, $num:Int!){
  organization(login:$org){ projectV2(number:$num){
    id title
    fields(first:30){ nodes{
      ... on ProjectV2FieldCommon { id name dataType }
      ... on ProjectV2SingleSelectField { id name options{ id name } }
    }}
  }}
}
```
Variables: `{"org": "AllenNeuralDynamics", "num": 184}`.

## Create an issue (REST)

```
POST /repos/AllenNeuralDynamics/<repo>/issues
{"title": "...", "body": "...", "labels": ["documentation", "priority:P1"]}
```
Response carries `number`, `html_url`, and `node_id` — the `node_id` is what step 2 takes.

## Add to the board

```graphql
mutation($p:ID!, $c:ID!){
  addProjectV2ItemById(input:{projectId:$p, contentId:$c}){ item{ id } } }
```
`$c` is the issue's `node_id`; the returned `item.id` (`PVTI_…`) is what field mutations
take. Idempotent — re-adding an existing item returns the same item id.

## Set a single-select field

One mutation per field.

```graphql
mutation($p:ID!, $i:ID!, $f:ID!, $o:String!){
  updateProjectV2ItemFieldValue(input:{
    projectId:$p, itemId:$i, fieldId:$f, value:{singleSelectOptionId:$o}
  }){ projectV2Item{ id } } }
```

Other field types swap the `value` member: `{text: "..."}`, `{number: 3}`,
`{date: "2026-09-01"}`, `{iterationId: "..."}`. A wrong-but-well-formed option id succeeds
silently, so always read back.

## Read an item back

```graphql
query($i:ID!){ node(id:$i){ ... on ProjectV2Item {
  content{ ... on Issue { number title url } }
  fieldValues(first:20){ nodes{
    ... on ProjectV2ItemFieldSingleSelectValue { name field{ ... on ProjectV2SingleSelectField { name } } }
  }}
}}}
```

## Gotchas

- **Item id vs issue node id.** Field mutations take the `PVTI_…` item id. Passing the
  `I_…` issue node id fails with a type error that does not name the confusion.
- **Errors arrive HTTP 200.** GraphQL puts failures in a top-level `errors` array with a
  200 status, so check `errors` explicitly — `urlopen` will not raise.
- **SAML SSO.** 403 "Resource protected by organization SAML enforcement" means the PAT is
  valid but not SSO-authorized for `AllenNeuralDynamics`. Authorize the token; do not retry.
- **Scopes.** `repo` to create issues, `project` to write board fields.
- Deleting an item from the board (`deleteProjectV2Item`) does not close or delete the
  issue, and closing an issue does not remove its board item — Status must be moved.

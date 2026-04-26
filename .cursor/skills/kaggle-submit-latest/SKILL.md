---
name: kaggle-submit-latest
description: Automates Kaggle competition submission for the latest local CSV in submissions/, then appends submission outcomes to experiment.log. Use when the user asks to submit latest Kaggle predictions, run Kaggle MCP submission flow, or wants upload+submit+status reporting without manual steps.
---

# Kaggle Submit Latest

## Goal

Submit the most recent CSV in `submissions/` to Kaggle via MCP, append results to `experiment.log`, and report final status/scores.

## Required Tools

- `ReadFile` for MCP tool schema files
- `CallMcpTool` for Kaggle MCP calls
- `Shell` for file metadata + upload (`curl.exe`)

## Workflow Checklist

Copy this checklist and complete every item:

```text
- [ ] 1) Verify latest submission CSV exists
- [ ] 2) Read Kaggle MCP schemas (mandatory before calling)
- [ ] 3) Start upload session via MCP
- [ ] 4) Upload CSV bytes to create_url
- [ ] 5) Finalize competition submission via MCP
- [ ] 6) Fetch submission status/details
- [ ] 7) Append structured submission record to experiment.log
- [ ] 8) Report ref, status, public/private scores
```

## Exact Execution Steps

### 1) Locate latest submission file

Use shell in repo root:

```powershell
ls "submissions"
```

Pick the latest `.csv` by modified time. Capture:
- full relative path (example: `submissions\submission_<run_id>_full.csv`)
- file size bytes
- `lastModifiedEpochSeconds`
- file name

PowerShell helper:

```powershell
$f = Get-ChildItem "submissions\*.csv" | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
Write-Output "$($f.FullName)|$($f.Name)|$($f.Length)|$([int][double]::Parse((Get-Date $f.LastWriteTimeUtc -UFormat %s)))"
```

### 2) Read MCP schemas first (mandatory)

Read these descriptor files before tool calls:

- `mcps/project-0-project_v2-kaggle/tools/start_competition_submission_upload.json`
- `mcps/project-0-project_v2-kaggle/tools/submit_to_competition.json`
- `mcps/project-0-project_v2-kaggle/tools/get_competition_submission.json`
- `mcps/project-0-project_v2-kaggle/tools/get_competition.json`

### 3) Start upload session

Call `start_competition_submission_upload`:

- `competitionName`: competition slug (default `home-credit-default-risk` unless user specifies another)
- `hasCompetitionName`: `true`
- `contentLength`: file bytes
- `lastModifiedEpochSeconds`: integer
- `fileName`: file basename

Store response:
- `token`
- `create_url`

### 4) Upload file bytes to `create_url`

Use `curl.exe` in PowerShell (not `Invoke-WebRequest`, which may hang):

```powershell
curl.exe -sS -o NUL -w "%{http_code}" -X PUT --upload-file "<RELATIVE_CSV_PATH>" "<CREATE_URL>"
```

Continue only if HTTP code is `200`.

### 5) Submit to competition

Call `submit_to_competition` with:

- `competitionName`
- `blobFileTokens` = upload `token`
- `submissionDescription` (brief text with run id/timestamp)
- `hasSubmissionDescription` = `true`

### 6) Fetch status

Use `get_competition_submission` with returned `ref` from submit step.

If status is not final yet, poll `get_competition_submission` briefly until terminal state or timeout.

### 7) Append result to `experiment.log`

After fetching final submission details, append one structured line to `experiment.log` via shell:

```powershell
$ts = (Get-Date).ToUniversalTime().ToString("o")
$line = "$ts - INFO - kaggle_submit - competition=<COMPETITION> file=<FILE_NAME> ref=<REF> status=<STATUS> public_score=<PUBLIC_OR_NA> private_score=<PRIVATE_OR_NA> description=\"<DESCRIPTION>\""
Add-Content -Path "experiment.log" -Value $line
```

Rules:
- Always log `competition`, `file`, `ref`, `status`.
- Use `NA` when score fields are missing.
- Keep it single-line for grep/search friendliness.

## Reporting Format

Return concise result:

- competition slug
- uploaded file path + size
- submission ref
- status
- public score (if present)
- private score (if present)
- submitted timestamp (if present)
- confirmation that `experiment.log` was updated

## Failure Handling

- If Kaggle MCP responds `Unauthenticated`: instruct user to set token header in `.cursor/mcp.json`, reload Cursor, and retry.
- If upload HTTP is not `200`: stop and report code + likely cause.
- If MCP server missing: report available servers and ask user to enable Kaggle MCP.
- Never log or echo full bearer tokens.

## Notes

- Do not ask the user to run manual steps if tooling can proceed.
- Prefer this automated flow over ad hoc manual Kaggle UI submissions.

$ErrorActionPreference = 'Stop'
Set-Location E:\Project

# 1) Commit working changes (.gitignore excludes .env, chroma_db, models, data/docs, *.log)
git add -A
git commit -q -m "KodeksBot: Giga-Embeddings-instruct-480M on GPU, irrelevant-distance threshold 1.35, collapsible sources UI

- config.py: model_dir -> Giga-Embeddings-instruct-480M-0826, embed_device=cuda, trust_remote_code
- rag/engine.py: InstructGigaEmbeddings, PDF via fitz, _MAX_IRRELEVANT_DISTANCE 0.60 -> 1.35
- static/index.html: sources collapse into a expandable link
- NOTES.md/AGENTS.md: docs for the new embedding model"
Write-Host "COMMITTED"

# 2) Token from Windows Credential Manager (do NOT print)
$cred = ("protocol=https`nhost=github.com`n" | git credential fill 2>$null)
$token = ($cred | Where-Object { $_ -match '^password=' }) -replace 'password=',''
if (-not $token) { $token = ($cred | Where-Object { $_ -match '^username=' }) -replace 'username=','' }
$headers = @{ Authorization = "Bearer $token"; 'User-Agent' = 'git-client' }

$me = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers
$login = $me.login
Write-Host "GH LOGIN=$login"

# 3) Create private repo
$repoName = 'kodeksbot'
$body = @{ name = $repoName; private = $true; description = 'KodeksBot - Russian legal RAG assistant (FastAPI+LangChain+Chroma+GigaChat) with local Giga embeddings' } | ConvertTo-Json
try {
    $r = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType 'application/json'
    Write-Host "REPO CREATED: $($r.html_url)"
} catch {
    $msg = $_.ErrorDetails.Message
    if ($msg -match 'already exist' -or $msg -match 'name already') { Write-Host "REPO EXISTS (reuse)" }
    else { Write-Host "REPO CREATE ERR: $msg"; throw }
}

# 4) Remote + push
$remoteUrl = "https://github.com/$login/$repoName.git"
if (-not (git remote get-url origin 2>$null)) {
    git remote add origin $remoteUrl
    Write-Host "REMOTE ADDED $remoteUrl"
} else {
    Write-Host "REMOTE EXISTS"
}
git push -u origin main 2>&1 | Out-String -Width 200 | Write-Host
Write-Host "PUSH DONE"

$env:STRATA_REQUIRE_SPIRV_VAL = '1'
$skipped = [System.Collections.Generic.List[string]]::new()

cargo test --workspace --locked -- --nocapture 2>&1 | ForEach-Object {
    $line = $_.ToString()
    if ($line -match '^SKIPPED\s' -or $line -match ': skipped\s*$') {
        $skipped.Add($line)
    }
    Write-Host $line
}
$status = $LASTEXITCODE
Write-Host "Reported hardware or tool skips: $($skipped.Count)"
$skipped | ForEach-Object { Write-Host "  $_" }
exit $status

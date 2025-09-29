
# Params
$datasetPath = ".\flower_photos"  
$outputDir = ".\labels"                
$trainRatio = 0.7                      
$valRatio = 0.2                       
# testRatio = 1 - trainRatio - valRatio

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$allImages = Get-ChildItem -Path $datasetPath -Recurse -File -Include *.jpg,*.png |
             Select-Object -ExpandProperty FullName |
             ForEach-Object { $_.Substring((Resolve-Path $datasetPath).Path.Length + 1) }

$random = New-Object System.Random
$shuffled = $allImages | Sort-Object { $random.Next() }

$totalCount = $shuffled.Count
$trainCount = [math]::Floor($totalCount * $trainRatio)
$valCount = [math]::Floor($totalCount * $valRatio)

$trainSet = $shuffled[0..($trainCount-1)]
$valSet = $shuffled[$trainCount..($trainCount+$valCount-1)]
$testSet = $shuffled[($trainCount+$valCount)..($totalCount-1)]

$trainSet | Out-File -FilePath "$outputDir\train.txt" -Encoding utf8
$valSet | Out-File -FilePath "$outputDir\val.txt" -Encoding utf8
$testSet | Out-File -FilePath "$outputDir\test.txt" -Encoding utf8

Write-Host "Separate Done："
Write-Host "Train: $($trainSet.Count)"
Write-Host "Val: $($valSet.Count)"
Write-Host "Test: $($testSet.Count)"

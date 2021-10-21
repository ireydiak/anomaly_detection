## Preprocessing pipeline

## Linux

## Windows
Found on StackOverflow[https://stackoverflow.com/questions/27892957/merging-multiple-csv-files-into-one-using-powershell]

```powershell
$getFirstLine = $true

get-childItem "*.csv" | foreach {
    $filePath = $_

    $lines =  $lines = Get-Content $filePath  
    $linesToWrite = switch($getFirstLine) {
           $true  {$lines}
           $false {$lines | Select -Skip 1}

    }

    $getFirstLine = $false
    Add-Content "all-flows.csv" $linesToWrite
    }
```
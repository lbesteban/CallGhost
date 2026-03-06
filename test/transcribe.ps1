param (
    [string]$Language = "en-US", # Change to "es-ES" for Spanish
    [string]$OutputFile = "$HOME\Desktop\stream.txt"
)

Add-Type -AssemblyName System.Speech

# 1. Setup Engine with specific language
try {
    $Culture = New-Object System.Globalization.CultureInfo($Language)
    $Engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($Culture)
} catch {
    Write-Warning "Language pack '$Language' not found. Falling back to default."
    $Engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
}

# 2. Setup Input (Listen to Default Recording Device -> Set this to CABLE Output)
$Engine.SetInputToDefaultAudioDevice()

# 3. Create a Dictation Grammar (Free speech)
$Grammar = New-Object System.Speech.Recognition.DictationGrammar
$Engine.LoadGrammar($Grammar)

# 4. Event Handler for "Real-time" writing
$EventHandler = {
    param($Sender, $Args)
    $Text = $Args.Result.Text
    $Timestamp = (Get-Date).ToString("HH:mm:ss")
    $Line = "[$Timestamp] $Text"
    
    # Write to console
    Write-Host $Line -ForegroundColor Cyan
    
    # Stream to file (Force append immediately)
    [System.IO.File]::AppendAllText($OutputFile, "$Line`r`n")
}

Register-ObjectEvent -InputObject $Engine -EventName SpeechRecognized -Action $EventHandler

# 5. Start
Write-Host "Listening to CABLE Output in ($Language)..." -ForegroundColor Green
Write-Host "Streaming to: $OutputFile" -ForegroundColor Gray
$Engine.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)

# Keep alive
while ($true) { Start-Sleep -Seconds 1 }

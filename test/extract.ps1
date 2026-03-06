# 1. Load the Speech Recognition library
Add-Type -AssemblyName System.Speech

# 2. Define the output text file path
$FilePath = "transcription.txt"
"--- New Transcription Session: $(Get-Date) ---" | Out-File -FilePath $FilePath -Append

# 3. Initialize the Speech Recognition Engine
$SpeechEngine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$SpeechEngine.SetInputToDefaultAudioDevice() # It will listen to VB-Audio Cable

# 4. Load a basic dictation grammar
$Grammar = New-Object System.Speech.Recognition.DictationGrammar
$SpeechEngine.LoadGrammar($Grammar)

# 5. Define what happens when speech is recognized
$Action = {
    param($Sender, $EventArgs)
    $Text = $EventArgs.Result.Text
    Write-Host "Recognized: $Text" -ForegroundColor Cyan
    $Text | Out-File -FilePath $FilePath -Append
}

# 6. Register the event to write to the file
Register-ObjectEvent -InputObject $SpeechEngine -EventName SpeechRecognized -Action $Action

# 7. Start listening indefinitely
$SpeechEngine.RecognizeAsync([System.Speech.Recognition.RecognizeMode]::Multiple)

Write-Host "Listening to VB-Audio Cable... Press Ctrl+C to stop." -ForegroundColor Green

# Keep the script alive
while($true) { Start-Sleep -Seconds 1 }

param (
    [string]$Language = "es-ES", 
    [string]$OutputFile = ".\out.txt"
)

Add-Type -AssemblyName System.Speech

# 1. Preparar el motor con tiento
try {
    $Culture = New-Object System.Globalization.CultureInfo($Language)
    $Engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($Culture)
} catch {
    Write-Warning "No encuentro el idioma $Language. Uso el de por defecto para no parar la trilla."
    $Engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine
}

# 2. Cargar la gramática (El punto conflictivo)
# Creamos el objeto primero para que no se atragante la línea
$Dictation = New-Object System.Speech.Recognition.DictationGrammar
$Engine.LoadGrammar($Dictation)

# 3. Configurar entrada
try {
    $Engine.SetInputToDefaultAudioDevice()
} catch {
    Write-Error "¡No veo ningún micro o cable conectado! Revisa el panel de sonido."
    exit
}

Write-Host "--- Iniciando escucha ---" -ForegroundColor Green
Write-Host "Idioma: $($Engine.RecognizerInfo.Culture.Name)" -ForegroundColor Gray
Write-Host "Pulsa 'ESC' para cerrar el chiringuito."

# 4. Bucle síncrono
while ($true) {
    if ([System.Console]::KeyAvailable) {
        $key = [System.Console]::ReadKey($true)
        if ($key.Key -eq "Escape") { break }
    }

    # Recognize() devuelve control cada vez que hay silencio o tras un timeout
    $Result = $Engine.Recognize([TimeSpan]::FromSeconds(1))

    if ($Result -ne $null -and $Result.Text -ne "") {
        $Timestamp = (Get-Date).ToString("HH:mm:ss")
        $Line = "[$Timestamp] $($Result.Text)"
        Write-Host $Line -ForegroundColor Cyan
        Add-Content -Path $OutputFile -Value $Line
    }
}

Write-Host "Cerrando y guardando en $OutputFile" -ForegroundColor Yellow
$Engine.Dispose()

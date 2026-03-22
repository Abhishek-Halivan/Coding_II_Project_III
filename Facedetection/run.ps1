param(
    [string]$CascadePath,
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Release"
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$vcpkgRoot = $env:VCPKG_ROOT
if (-not $vcpkgRoot -and (Test-Path "C:\vcpkg")) {
    $vcpkgRoot = "C:\vcpkg"
}

$toolchainFile = $null
if ($vcpkgRoot) {
    $toolchainFile = Join-Path $vcpkgRoot "scripts\buildsystems\vcpkg.cmake"
}

$exePath = Join-Path $projectRoot "build\$BuildType\face_detection.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "Binary not found. Configuring and building first..."

    if ($toolchainFile -and (Test-Path $toolchainFile)) {
        cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE="$toolchainFile" -DVCPKG_TARGET_TRIPLET=x64-windows
    } else {
        cmake -S . -B build
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configure failed."
        exit $LASTEXITCODE
    }

    cmake --build build --config $BuildType
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed."
        exit $LASTEXITCODE
    }
}

$dnnPbUrl = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb"
$dnnPbtxtUrl = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"

if (-not (Test-Path "$projectRoot\opencv_face_detector_uint8.pb")) {
    Write-Host "Downloading OpenCV DNN face detector weights (~2.6 MB)..."
    Invoke-WebRequest -Uri $dnnPbUrl -OutFile "$projectRoot\opencv_face_detector_uint8.pb"
}

if (-not (Test-Path "$projectRoot\opencv_face_detector_uint8.pbtxt")) {
    Write-Host "Downloading OpenCV DNN face detector config..."
    Invoke-WebRequest -Uri $dnnPbtxtUrl -OutFile "$projectRoot\opencv_face_detector_uint8.pbtxt"
}

if ($vcpkgRoot) {
    $vcpkgBin = Join-Path $vcpkgRoot "installed\x64-windows\bin"
    if (Test-Path $vcpkgBin) {
        $env:Path = "$vcpkgBin;$env:Path"
    }
}

if (-not $CascadePath) {
    if ($env:OPENCV_FACE_CASCADE -and (Test-Path $env:OPENCV_FACE_CASCADE)) {
        $CascadePath = $env:OPENCV_FACE_CASCADE
    } elseif ($vcpkgRoot) {
        $CascadePath = Get-ChildItem -Path $vcpkgRoot -Filter "haarcascade_frontalface_default.xml" -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }

    if (-not $CascadePath) {
        $localCascade = Join-Path $projectRoot "haarcascade_frontalface_default.xml"
        if (Test-Path $localCascade) {
            $CascadePath = $localCascade
        }
    }
}

if ($CascadePath) {
    Write-Host "Using cascade: $CascadePath"
    & $exePath $CascadePath
} else {
    Write-Host "No cascade path found automatically. Running without args."
    & $exePath
}

exit $LASTEXITCODE

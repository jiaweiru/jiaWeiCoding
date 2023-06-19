@echo off
setlocal enabledelayedexpansion

set "source1=文件夹路径1"
set "source2=文件夹路径2"
set "command=peaq -r"
set "decodedSuffix=decoded"

for %%F in ("%source1%\*") do (
    set "filename=%%~nF"
    set "extension=%%~xF"
    set "decodedFilename=!filename!%decodedSuffix%!extension!"

    if exist "%source2%\!decodedFilename!" (
        %command% "%%F" -t "%source2%\!decodedFilename!"
    ) else (
        echo "%source2%\!decodedFilename!" 文件不存在
    )
)

endlocal
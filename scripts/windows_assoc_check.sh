#!/usr/bin/env bash
set -euo pipefail

glview_exe="${1:-C:\\path\\to\\glview.exe}"
test_image="${2:-C:\\path\\to\\test.png}"

cat <<EOF
WSL helper for checking glview Windows file associations

1. Register glview as a Windows handler:
   /mnt/c/Windows/System32/cmd.exe /c "$glview_exe --install-default-handler"

2. Inspect RegisteredApplications and Capabilities registration:
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\RegisteredApplications"
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\glview\\Capabilities" /s

3. Inspect the application registration:
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\Classes\\Applications\\glview.exe" /s

4. Inspect per-extension OpenWith registration:
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\Classes\\.png\\OpenWithProgids"
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\Classes\\.jpg\\OpenWithProgids"

5. Open the glview-specific Default Apps settings page directly:
   /mnt/c/Windows/System32/cmd.exe /c start "" "ms-settings:defaultapps?registeredAppUser=glview"

6. Inspect the current default-app choice:
   /mnt/c/Windows/System32/reg.exe query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\.png\\UserChoice"

7. Launch a file through normal Windows shell resolution:
   /mnt/c/Windows/System32/cmd.exe /c start "" "$test_image"

Notes:
 - Steps 2-4 verify that glview registered itself correctly for both Default Apps and Open With.
 - Step 5 should land on glview's page in Windows Default Apps on supported Windows 11 builds.
 - Step 6 may still point to another app until the user confirms glview in Windows Settings.
 - Step 7 is the real shell-level check: if glview is the current default for that extension, it should open there.
EOF

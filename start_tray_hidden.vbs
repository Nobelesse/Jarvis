Option Explicit

Dim shell
Dim fileSystem
Dim projectFolder
Dim pythonwPath
Dim trayPath
Dim command

Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")

projectFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)

pythonwPath = projectFolder & "\.venv\Scripts\pythonw.exe"
trayPath = projectFolder & "\tray_app.py"

If Not fileSystem.FileExists(pythonwPath) Then
    MsgBox "Jarvis virtual environment was not found.", 16, "Jarvis"
    WScript.Quit 1
End If

If Not fileSystem.FileExists(trayPath) Then
    MsgBox "tray_app.py was not found.", 16, "Jarvis"
    WScript.Quit 1
End If

command = """" & pythonwPath & """" & " " & """" & trayPath & """"

shell.Run command, 0, False
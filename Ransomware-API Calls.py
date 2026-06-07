import os
import re
import pandas as pd

API_Calls = [
    "CreateFile", "ReadFile", "WriteFile", "CloseHandle", "DeleteFile",
    "GetFileAttributes", "SetFileAttributes", "FindFirstFile", "FindNextFile",
    "CopyFile", "MoveFile", "OpenFile",
    "RegOpenKey", "RegSetValue", "RegQueryValue", "RegCreateKey", "RegCloseKey",
    "RegDeleteKey", "RegEnumKey", "RegEnumValue", "RegDeleteValue",
    "WinHttpConnect", "WinHttpOpenRequest", "WinHttpSendRequest", "WinHttpReceiveResponse",
    "send", "recv", "bind", "listen", "connect", "socket", "closesocket",
    "VirtualAlloc", "VirtualFree", "GlobalAlloc", "GlobalFree", "HeapAlloc", "HeapFree",
    "CreateProcess", "TerminateProcess", "GetExitCodeProcess", "OpenProcess",
    "CreateThread", "ExitThread", "WaitForSingleObject", "WaitForMultipleObjects",
    "TerminateThread", "ResumeThread", "SuspendThread",
    "EnterCriticalSection", "LeaveCriticalSection",
    "CreateMutex", "ReleaseMutex",
    "CreateEvent", "SetEvent", "ResetEvent",
    "CryptAcquireContext", "CryptEncrypt", "CryptDecrypt", "CryptGenKey", "CryptDestroyKey",
    "OpenService", "StartService", "ControlService", "DeleteService", "CloseServiceHandle",
    "FindWindow", "ShowWindow", "SetWindowText", "GetWindowText", "GetWindowRect",
    "SetForegroundWindow", "GetLastError", "SetLastError",
    "getFile", "write", "encrypt", "createWritable", "showDirectoryPicker",
    "FileSystemFileHandle", "chooseFileSystemEntries",
    "ReadDirectoryChangesW", "GetFileSize", "SetFilePointer", "FlushFileBuffers",
    "LockFile", "UnlockFile", "BackupRead", "BackupWrite", "ReplaceFile",
    "FindClose", "GetFileTime", "SetFileTime",
    "RegFlushKey", "RegSaveKey", "RegLoadKey", "RegLoadMUIString",
    "RegRestoreKey", "RegNotifyChangeKeyValue",
    "WSAStartup", "WSACleanup", "WSASend", "WSARecv",
    "InternetOpen", "InternetCloseHandle", "InternetReadFile", "InternetWriteFile",
    "InternetSetOption", "HttpSendRequest", "HttpQueryInfo", "GetHostByName",
    "GetHostByAddr", "GetAddrInfo", "FreeAddrInfo", "Socket",
    "MapViewOfFile", "UnmapViewOfFile", "VirtualProtect", "VirtualQuery",
    "GlobalLock", "GlobalUnlock", "LocalAlloc", "LocalFree", "GlobalReAlloc",
    "VirtualLock", "VirtualUnlock", "RtlMoveMemory",
    "ShellExecute", "ShellExecuteEx",
    "GetProcessTimes", "GetThreadTimes",
    "CreateRemoteThread", "OpenThread",
    "SuspendProcess", "ResumeProcess",
    "GetProcessId", "ExitProcess", "OpenJobObject",
    "CreateSemaphore", "ReleaseSemaphore",
    "SetWaitableTimer", "CancelWaitableTimer",
    "InterlockedIncrement", "InterlockedDecrement", "PulseEvent",
    "InitializeCriticalSectionAndSpinCount",
    "CryptExportKey", "CryptImportKey", "CryptHashData", "CryptDeriveKey",
    "EncryptFile", "DecryptFile", "CryptProtectData", "CryptUnprotectData",
    "CryptHashSessionKey", "CryptSignHash", "CryptVerifySignature", "CryptSetKeyParam",
    "MoveWindow", "SetWindowLong", "GetWindowLong",
    "InvalidateRect", "RedrawWindow", "BringWindowToTop",
    "CreateWindowEx", "GetWindowThreadProcessId",
    "SendMessage", "PostMessage", "GetClassName", "AdjustWindowRect",
    "GetSystemMetrics", "GetSystemTime", "SetSystemTime",
    "QueryPerformanceCounter", "QueryPerformanceFrequency", "GetTickCount",
    "GetDevicePowerState",
    "FormatMessage", "SetUnhandledExceptionFilter", "RaiseException",
    "CreatePipe", "CreateNamedPipe", "ConnectNamedPipe", "DisconnectNamedPipe",
    "IsDebuggerPresent", "DebugActiveProcess", "DebugBreak"
]

dataset = "/Downloads/theZoo-master/malware/Source/Original"
results = []

for i in os.listdir(dataset):
    folder_path = os.path.join(dataset, i)

    if os.path.isdir(folder_path):
        count = {api: 0 for api in API_Calls}

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for api in API_Calls:
                    count[api] += len(re.findall(api, content))

        result = {'Ransomware': i}
        result.update(count)
        results.append(result)

df = pd.DataFrame(results)

output_file = "API Calls.xlsx"
df.to_excel(output_file, index=False)


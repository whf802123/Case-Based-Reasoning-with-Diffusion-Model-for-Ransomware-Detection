
import os
import re
import pandas as pd

dataset = "/Downloads/theZoo-master/malware/Source/Original"

api_categories = {
    "File Operation APIs": [
        "CreateFile", "ReadFile", "WriteFile", "DeleteFile", "GetFileAttributes",
        "SetFileAttributes", "FindFirstFile", "FindNextFile", "CopyFile", "MoveFile",
        "OpenFile", "FlushFileBuffers", "LockFile", "UnlockFile", "BackupRead",
        "BackupWrite", "ReplaceFile", "FindClose", "GetFileSize", "SetFilePointer",
        "getFile", "write", "createWritable", "showDirectoryPicker",
        "FileSystemFileHandle", "chooseFileSystemEntries"
    ],
    "Network Operation APIs": [
        "send", "recv", "bind", "listen", "connect", "socket", "closesocket",
        "WinHttpConnect", "WinHttpOpenRequest", "WinHttpSendRequest", "WinHttpReceiveResponse",
        "WSAStartup", "WSACleanup", "WSASend", "WSARecv",
        "InternetOpen", "InternetCloseHandle", "InternetReadFile", "InternetWriteFile",
        "InternetSetOption", "HttpSendRequest", "HttpQueryInfo", "GetHostByName",
        "GetHostByAddr", "GetAddrInfo", "FreeAddrInfo", "Socket"
    ],
    "Memory Management APIs": [
        "VirtualAlloc", "VirtualFree", "HeapAlloc", "HeapFree",
        "GlobalAlloc", "GlobalFree", "GlobalLock", "GlobalUnlock",
        "LocalAlloc", "LocalFree", "GlobalReAlloc",
        "VirtualLock", "VirtualUnlock", "RtlMoveMemory",
        "MapViewOfFile", "UnmapViewOfFile", "VirtualProtect", "VirtualQuery"
    ],
    "Registry Operation APIs": [
        "RegOpenKey", "RegSetValue", "RegQueryValue", "RegCreateKey", "RegCloseKey",
        "RegDeleteKey", "RegEnumKey", "RegEnumValue", "RegDeleteValue",
        "RegFlushKey", "RegSaveKey", "RegLoadKey", "RegLoadMUIString",
        "RegRestoreKey", "RegNotifyChangeKeyValue"
    ],
    "Process and Thread Management APIs": [
        "CreateProcess", "TerminateProcess", "GetExitCodeProcess", "OpenProcess",
        "CreateThread", "ExitThread", "WaitForSingleObject", "WaitForMultipleObjects",
        "TerminateThread", "ResumeThread", "SuspendThread",
        "CreateRemoteThread", "OpenThread", "SuspendProcess", "ResumeProcess",
        "GetProcessId", "ExitProcess", "OpenJobObject"
    ],
    "Encryption APIs": [
        "CryptAcquireContext", "CryptEncrypt", "CryptDecrypt", "CryptGenKey", "CryptDestroyKey",
        "CryptExportKey", "CryptImportKey", "CryptHashData", "CryptDeriveKey",
        "EncryptFile", "DecryptFile", "CryptProtectData", "CryptUnprotectData",
        "CryptHashSessionKey", "CryptSignHash", "CryptVerifySignature", "CryptSetKeyParam",
        "encrypt"
    ],
    "GUI APIs": [
        "FindWindow", "ShowWindow", "SetWindowText", "GetWindowText", "GetWindowRect",
        "SetForegroundWindow", "MoveWindow", "SetWindowLong", "GetWindowLong",
        "InvalidateRect", "RedrawWindow", "BringWindowToTop", "CreateWindowEx",
        "GetWindowThreadProcessId", "SendMessage", "PostMessage", "GetClassName", "AdjustWindowRect"
    ],
    "System Management APIs": [
        "GetSystemMetrics", "GetSystemTime", "SetSystemTime",
        "QueryPerformanceCounter", "QueryPerformanceFrequency", "GetTickCount",
        "GetDevicePowerState", "GetLastError", "SetLastError",
        "FormatMessage", "SetUnhandledExceptionFilter", "RaiseException"
    ],
    "Pipe APIs": [
        "CreatePipe", "CreateNamedPipe", "ConnectNamedPipe", "DisconnectNamedPipe"
    ],
    "Debug APIs": [
        "IsDebuggerPresent", "DebugActiveProcess", "DebugBreak"
    ]
}

all_apis = []
for apis in api_categories.values():
    for api in apis:
        if api not in all_apis:
            all_apis.append(api)

results = []

for folder_name in os.listdir(dataset):
    folder_path = os.path.join(dataset, folder_name)

    if os.path.isdir(folder_path):
        api_count = {api: 0 for api in all_apis}

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except:
                    continue
                for api in all_apis:
                    api_count[api] += len(re.findall(r"\b" + re.escape(api) + r"\b", content))

        result = {"Ransomware": folder_name}

        for category, apis in api_categories.items():
            result[category + " Total"] = sum(api_count[api] for api in apis)
            for api in apis:
                result[api] = api_count[api]

        results.append(result)

columns = ["Ransomware"]

for category, apis in api_categories.items():
    columns.append(category + " Total")
    columns.extend(apis)

df = pd.DataFrame(results)
df = df[columns]

output_file = "API_Categories_and_Individual_Counts.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")


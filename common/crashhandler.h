#pragma once

// Crash handler: emits a stack trace and writes a minidump on Windows; noop on other platforms.
//
// Usage:
//   #include "crashhandler.h"
//   int main() { crashhandler_init(); ... }

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <dbghelp.h>

#include <cstdio>
#include <ctime>

#pragma comment(lib, "dbghelp.lib")

static void crashhandler_write_minidump(EXCEPTION_POINTERS * exc_info) {
    char dump_path[MAX_PATH];
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_s(&tm_buf, &now);
    snprintf(dump_path, sizeof(dump_path), "crash-%04d%02d%02d-%02d%02d%02d-%lu.dmp",
             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
             (unsigned long)GetCurrentProcessId());

    HANDLE file = CreateFileA(dump_path, GENERIC_WRITE, 0, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "\nFailed to create minidump file: %s (error %lu)\n",
                dump_path, GetLastError());
        return;
    }

    MINIDUMP_EXCEPTION_INFORMATION mei;
    mei.ThreadId          = GetCurrentThreadId();
    mei.ExceptionPointers = exc_info;
    mei.ClientPointers    = FALSE;

    // MiniDumpWithDataSegs includes global variable contents which helps debugging
    BOOL ok = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                                MiniDumpWithDataSegs, &mei, nullptr, nullptr);
    CloseHandle(file);

    if (ok) {
        fprintf(stderr, "\nMinidump written to: %s\n", dump_path);
    } else {
        fprintf(stderr, "\nFailed to write minidump (error %lu)\n", GetLastError());
    }
}

static void crashhandler_print_stacktrace(CONTEXT * ctx) {
    HANDLE process = GetCurrentProcess();
    HANDLE thread  = GetCurrentThread();

    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(process, nullptr, TRUE);

    STACKFRAME64 frame = {};
    DWORD machine_type;

#if defined(_M_X64) || defined(__x86_64__)
    machine_type           = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset    = ctx->Rip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = ctx->Rbp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = ctx->Rsp;
    frame.AddrStack.Mode   = AddrModeFlat;
#elif defined(_M_ARM64) || defined(__aarch64__)
    machine_type           = IMAGE_FILE_MACHINE_ARM64;
    frame.AddrPC.Offset    = ctx->Pc;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = ctx->Fp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = ctx->Sp;
    frame.AddrStack.Mode   = AddrModeFlat;
#elif defined(_M_IX86) || defined(__i386__)
    machine_type           = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset    = ctx->Eip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = ctx->Ebp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = ctx->Esp;
    frame.AddrStack.Mode   = AddrModeFlat;
#else
    // Unsupported architecture — skip stack walk
    fprintf(stderr, "(stack walk not supported on this architecture)\n");
    SymCleanup(process);
    return;
#endif

    // Buffer for symbol name resolution
    alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    SYMBOL_INFO * symbol  = reinterpret_cast<SYMBOL_INFO *>(sym_buf);
    symbol->SizeOfStruct  = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen    = MAX_SYM_NAME;

    IMAGEHLP_LINE64 line  = {};
    line.SizeOfStruct     = sizeof(IMAGEHLP_LINE64);

    fprintf(stderr, "\nStack trace:\n");

    for (int i = 0; i < 64; i++) {
        if (!StackWalk64(machine_type, process, thread, &frame, ctx,
                         nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) {
            break;
        }

        fprintf(stderr, "  #%-3d 0x%016llx", i, (unsigned long long)frame.AddrPC.Offset);

        DWORD64 sym_displacement = 0;
        if (SymFromAddr(process, frame.AddrPC.Offset, &sym_displacement, symbol)) {
            fprintf(stderr, " %s+0x%llx", symbol->Name, (unsigned long long)sym_displacement);
        }

        DWORD line_displacement = 0;
        if (SymGetLineFromAddr64(process, frame.AddrPC.Offset, &line_displacement, &line)) {
            fprintf(stderr, " (%s:%lu)", line.FileName, line.LineNumber);
        }

        fprintf(stderr, "\n");
    }

    SymCleanup(process);
}

static LONG WINAPI crashhandler_exception_handler(EXCEPTION_POINTERS * exc_info) {
    fprintf(stderr, "\n=== CRASH (unhandled exception) ===\n");
    fprintf(stderr, "Exception code:    0x%08lX\n", exc_info->ExceptionRecord->ExceptionCode);
    fprintf(stderr, "Exception address: 0x%p\n",    exc_info->ExceptionRecord->ExceptionAddress);

    crashhandler_print_stacktrace(exc_info->ContextRecord);
    crashhandler_write_minidump(exc_info);

    // Let the default handler (or a debugger) take over
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void crashhandler_init(void) {
    SetUnhandledExceptionFilter(crashhandler_exception_handler);
}

#else // !_WIN32

inline void crashhandler_init(void) {
    // noop on non-Windows platforms
}

#endif // _WIN32

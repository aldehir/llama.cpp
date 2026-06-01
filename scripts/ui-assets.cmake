# Provision UI assets and generate ui.cpp/ui.h.
#
# Asset provisioning priority:
#   1. Pre-built assets in SRC_DIST_DIR (manually placed in tools/ui/dist)
#   2. Prebuilt assets from the HF Bucket matching the build version
#      (when HF_ENABLED=ON)
#   3. If no versioned candidate is found, fall back based on BUILD_UI:
#        - BUILD_UI=OFF (default): pull 'latest' from the HF Bucket
#        - BUILD_UI=ON: build locally via 'npm run build'
#          (run 'npm install' beforehand)

cmake_minimum_required(VERSION 3.16)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(LLAMA_SOURCE_DIR  "" CACHE STRING "Project source root (to resolve version from git)")
set(HF_BUCKET         "" CACHE STRING "Hugging Face bucket name")
set(HF_VERSION        "" CACHE STRING "Version to download (empty = resolve from git)")
set(HF_ENABLED        "" CACHE STRING "Whether to allow HF Bucket download (ON/OFF)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(LLAMA_UI_EMBED    "" CACHE STRING "Path to llama-ui-embed helper")

set(ASSETS
    bundle.css
    bundle.js
    index.html
    loading.html
)

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(SRC_DIST_DIR "${UI_SOURCE_DIR}/dist")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(assets_present out_var)
    set(present TRUE)
    foreach(asset ${ASSETS})
        if(NOT EXISTS "${DIST_DIR}/${asset}")
            set(present FALSE)
            break()
        endif()
    endforeach()
    set(${out_var} ${present} PARENT_SCOPE)
endfunction()

function(copy_src_dist out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    foreach(asset ${ASSETS})
        if(NOT EXISTS "${SRC_DIST_DIR}/${asset}")
            return()
        endif()
    endforeach()

    file(MAKE_DIRECTORY "${DIST_DIR}")
    message(STATUS "UI: using pre-built assets from ${SRC_DIST_DIR}")
    foreach(asset ${ASSETS})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SRC_DIST_DIR}/${asset}" "${DIST_DIR}/${asset}"
        )
    endforeach()
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(npm_build_should_skip out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    assets_present(present)
    if(NOT present)
        return()
    endif()

    if(EXISTS "${STAMP_FILE}")
        return()
    endif()

    if(NOT EXISTS "${UI_SOURCE_DIR}/sources.cmake")
        return()
    endif()
    include("${UI_SOURCE_DIR}/sources.cmake")

    set(globs "")
    foreach(g ${UI_SOURCE_GLOBS})
        list(APPEND globs "${UI_SOURCE_DIR}/${g}")
    endforeach()
    file(GLOB_RECURSE sources ${globs})
    foreach(f ${UI_SOURCE_FILES})
        list(APPEND sources "${UI_SOURCE_DIR}/${f}")
    endforeach()

    file(TIMESTAMP "${DIST_DIR}/index.html" out_ts)

    foreach(s ${sources})
        if(NOT EXISTS "${s}")
            continue()
        endif()
        file(TIMESTAMP "${s}" s_ts)
        if(s_ts STRGREATER out_ts)
            return()
        endif()
    endforeach()

    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(npm_build out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${UI_SOURCE_DIR}/package.json")
        message(STATUS "UI: ${UI_SOURCE_DIR}/package.json not found, skipping npm")
        return()
    endif()

    npm_build_should_skip(skip)
    if(skip)
        message(STATUS "UI: npm output up-to-date, skipping build")
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_HOST_WIN32)
        find_program(NPM_EXECUTABLE NAMES npm.cmd npm.bat npm)
    else()
        find_program(NPM_EXECUTABLE npm)
    endif()
    if(NOT NPM_EXECUTABLE)
        message(STATUS "UI: npm not found, skipping npm build")
        return()
    endif()

    if(NOT EXISTS "${UI_SOURCE_DIR}/node_modules")
        message(STATUS "UI: node_modules not found in ${UI_SOURCE_DIR}; "
                       "run 'npm install' there before building with LLAMA_BUILD_UI=ON")
        return()
    endif()

    file(MAKE_DIRECTORY "${DIST_DIR}")

    message(STATUS "UI: running npm run build, output -> ${DIST_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "LLAMA_UI_OUT_DIR=${DIST_DIR}"
                ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY "${UI_SOURCE_DIR}"
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
        message(STATUS "UI: npm run build failed (${rc})")
        message(STATUS "  stderr: ${err}")
        return()
    endif()

    assets_present(present)
    if(NOT present)
        message(STATUS "UI: npm build finished but assets missing in ${DIST_DIR}")
        return()
    endif()

    message(STATUS "UI: npm build succeeded")
    # .ui-stamp records the HF ref of the last download; clear it so a later HF
    # run does not mistake these freshly built assets for a stamped download.
    file(REMOVE "${STAMP_FILE}")
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(resolve_version out_var)
    if(NOT "${HF_VERSION}" STREQUAL "")
        set(${out_var} "${HF_VERSION}" PARENT_SCOPE)
        return()
    endif()

    if(EXISTS "${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        include("${LLAMA_SOURCE_DIR}/cmake/build-info.cmake")
        if(NOT "${BUILD_NUMBER}" STREQUAL "" AND NOT BUILD_NUMBER EQUAL 0)
            set(${out_var} "b${BUILD_NUMBER}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(hf_download ref out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    file(MAKE_DIRECTORY "${DIST_DIR}")

    set(base "https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve/${ref}")

    message(STATUS "UI: downloading from ${ref}: ${base}")

    set(ok TRUE)
    foreach(asset ${ASSETS})
        file(DOWNLOAD "${base}/${asset}?download=true" "${DIST_DIR}/${asset}"
            STATUS status TIMEOUT 60
        )
        list(GET status 0 rc)
        if(NOT rc EQUAL 0)
            list(GET status 1 errmsg)
            message(STATUS "UI: download ${asset} from ${ref} failed: ${errmsg}")
            set(ok FALSE)
            break()
        endif()
        message(STATUS "UI: downloaded ${asset}")
    endforeach()

    if(NOT ok)
        return()
    endif()

    # Best-effort checksum verification
    file(DOWNLOAD "${base}/checksums.txt?download=true" "${DIST_DIR}/checksums.txt"
        STATUS cs_status TIMEOUT 30
    )
    list(GET cs_status 0 cs_rc)
    if(cs_rc EQUAL 0)
        message(STATUS "UI: verifying checksums")
        file(STRINGS "${DIST_DIR}/checksums.txt" cs_lines)
        foreach(asset ${ASSETS})
            file(SHA256 "${DIST_DIR}/${asset}" h)
            string(TOLOWER "${h}" h)
            string(REGEX MATCH "${h}[ \t]+${asset}" m "${cs_lines}")
            if(NOT m)
                message(WARNING "UI: checksum verification failed for ${asset}")
                set(ok FALSE)
                break()
            endif()
        endforeach()
        if(ok)
            message(STATUS "UI: all checksums verified")
        endif()
    endif()

    if(ok)
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(emit_files)
    assets_present(present)

    set(args "${UI_CPP}" "${UI_H}")
    if(present)
        foreach(asset ${ASSETS})
            list(APPEND args "${asset}" "${DIST_DIR}/${asset}")
        endforeach()
    endif()

    execute_process(
        COMMAND "${LLAMA_UI_EMBED}" ${args}
        RESULT_VARIABLE rc
    )
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "UI: llama-ui-embed failed (${rc})")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# 1. Priority 1: pre-built assets supplied in tools/ui/dist
# ---------------------------------------------------------------------------
copy_src_dist(SRC_OK)
if(SRC_OK)
    emit_files()
    return()
endif()

# ---------------------------------------------------------------------------
# 2. Priority 2: prebuilt assets from HF matching the build version.
#    Always attempted when HF_ENABLED=ON, regardless of BUILD_UI - a versioned
#    prebuilt is the "suitable candidate" for this exact build.
# ---------------------------------------------------------------------------
set(provisioned FALSE)

if(HF_ENABLED)
    resolve_version(VERSION)
    if(NOT "${VERSION}" STREQUAL "")
        set(stamp_ok FALSE)
        if(EXISTS "${STAMP_FILE}")
            file(READ "${STAMP_FILE}" stamped)
            string(STRIP "${stamped}" stamped)
            if("${stamped}" STREQUAL "${VERSION}")
                set(stamp_ok TRUE)
            endif()
        endif()

        assets_present(have_assets)
        if(stamp_ok AND have_assets)
            message(STATUS "UI: HF stamp '${stamped}' matches version, skipping HF fetch")
            set(provisioned TRUE)
        else()
            hf_download("${VERSION}" HF_OK)
            if(HF_OK)
                file(WRITE "${STAMP_FILE}" "${VERSION}")
                message(STATUS "UI: HF download succeeded, stamp updated (${VERSION})")
                set(provisioned TRUE)
            else()
                message(STATUS "UI: no prebuilt assets for version '${VERSION}'")
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Priority 3: no versioned candidate - fall back based on BUILD_UI.
#      BUILD_UI=ON  -> build locally via npm (no HF latest fallback)
#      BUILD_UI=OFF -> pull 'latest' from the HF Bucket (if HF_ENABLED=ON)
# ---------------------------------------------------------------------------
if(NOT provisioned)
    if(BUILD_UI)
        npm_build(NPM_OK)
        if(NPM_OK)
            set(provisioned TRUE)
        endif()
    elseif(HF_ENABLED)
        hf_download("latest" HF_OK)
        if(HF_OK)
            file(WRITE "${STAMP_FILE}" "latest")
            message(STATUS "UI: HF download succeeded, stamp updated (latest)")
            set(provisioned TRUE)
        else()
            message(STATUS "UI: HF download (latest) failed")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 4. Fallback: warn about stale or missing assets, then emit whatever we have
# ---------------------------------------------------------------------------
if(NOT provisioned)
    assets_present(have_assets)
    if(have_assets)
        message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
    else()
        message(WARNING "UI: no assets available - building without an embedded UI. "
                        "In a disconnected environment, download the pre-built UI "
                        "from a llama.cpp release at "
                        "https://github.com/ggml-org/llama.cpp/releases and "
                        "extract to tools/ui/dist.")
    endif()
endif()

emit_files()

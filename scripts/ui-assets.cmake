# Provision UI assets and generate ui.cpp/ui.h.
#
# Asset provisioning priority:
#   1. Pre-built assets in SRC_DIST_DIR (manually built by user)
#   2. If GH_ENABLED=ON: GitHub release download of the resolved version
#      (llama-<version>-ui.tar.gz)
#   3. If the above did not produce assets:
#        - BUILD_UI=ON:  npm build (npm ci runs as needed in the staged copy)
#        - BUILD_UI=OFF: GitHub release download of the latest release

cmake_minimum_required(VERSION 3.19)

set(UI_SOURCE_DIR     "" CACHE STRING "UI source directory (to run npm build)")
set(UI_BINARY_DIR     "" CACHE STRING "UI binary directory (to store generated files)")
set(LLAMA_SOURCE_DIR  "" CACHE STRING "Project source root (to resolve version from git)")
set(GH_REPO           "" CACHE STRING "GitHub repository (owner/repo) with UI release assets")
set(UI_VERSION        "" CACHE STRING "Release tag to download (empty = resolve from git)")
set(GH_ENABLED        "" CACHE STRING "Whether to allow GitHub release download (ON/OFF)")
set(BUILD_UI          "" CACHE STRING "Build UI via npm (ON/OFF)")
set(LLAMA_UI_EMBED    "" CACHE STRING "Path to llama-ui-embed helper")
set(LLAMA_UI_GZIP     "" CACHE STRING "Apply gzip compress to assets to save bandwidth")

set(DIST_DIR     "${UI_BINARY_DIR}/dist")
set(SRC_DIST_DIR "${UI_SOURCE_DIR}/dist")
set(WORK_DIR     "${UI_BINARY_DIR}/ui-src")
set(STAMP_FILE   "${UI_BINARY_DIR}/.ui-stamp")
set(UI_CPP       "${UI_BINARY_DIR}/ui.cpp")
set(UI_H         "${UI_BINARY_DIR}/ui.h")

function(npm_build_should_skip out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    if(NOT EXISTS "${DIST_DIR}/index.html")
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

function(stage_sources)
    if(EXISTS "${WORK_DIR}")
        file(GLOB staged RELATIVE "${WORK_DIR}" "${WORK_DIR}/*")
        list(REMOVE_ITEM staged "node_modules")
        foreach(entry ${staged})
            file(REMOVE_RECURSE "${WORK_DIR}/${entry}")
        endforeach()
    endif()

    file(COPY "${UI_SOURCE_DIR}/"
        DESTINATION "${WORK_DIR}"
        NO_SOURCE_PERMISSIONS
        PATTERN "node_modules" EXCLUDE
    )
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

    stage_sources()

    # npm writes node_modules/.package-lock.json on every successful install,
    # so a package-lock.json newer than this marker means node_modules is stale
    set(NPM_MARKER "${WORK_DIR}/node_modules/.package-lock.json")
    set(need_install FALSE)
    if(NOT EXISTS "${NPM_MARKER}")
        set(need_install TRUE)
    else()
        file(TIMESTAMP "${WORK_DIR}/package-lock.json" lock_ts)
        file(TIMESTAMP "${NPM_MARKER}" marker_ts)
        if(lock_ts STRGREATER marker_ts)
            set(need_install TRUE)
        endif()
    endif()

    if(need_install)
        message(STATUS "UI: running npm ci")
        execute_process(
            COMMAND ${NPM_EXECUTABLE} ci
            WORKING_DIRECTORY "${WORK_DIR}"
            RESULT_VARIABLE rc
            ERROR_VARIABLE  err
        )
        if(NOT rc EQUAL 0)
            message(STATUS "UI: npm ci failed (${rc})")
            message(STATUS "  stderr: ${err}")
            return()
        endif()
    endif()

    file(MAKE_DIRECTORY "${DIST_DIR}")

    message(STATUS "UI: running npm run build, output -> ${DIST_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "LLAMA_UI_OUT_DIR=${DIST_DIR}" "LLAMA_UI_VERSION=${UI_VERSION}" "LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
                ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY "${WORK_DIR}"
        RESULT_VARIABLE rc
        ERROR_VARIABLE  err
    )
    if(NOT rc EQUAL 0)
        message(STATUS "UI: npm run build failed (${rc})")
        message(STATUS "  stderr: ${err}")
        return()
    endif()

    if(NOT EXISTS "${DIST_DIR}/index.html")
        message(STATUS "UI: npm build finished but assets missing in ${DIST_DIR}")
        return()
    endif()

    message(STATUS "UI: npm build succeeded")
    file(REMOVE "${STAMP_FILE}")
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(resolve_version out_var)
    if(NOT "${UI_VERSION}" STREQUAL "")
        set(${out_var} "${UI_VERSION}" PARENT_SCOPE)
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

# Use GH_TOKEN to benefit from higher rate limits
function(gh_auth_headers out_var)
    set(${out_var} "" PARENT_SCOPE)
    if(DEFINED ENV{GH_TOKEN} AND NOT "$ENV{GH_TOKEN}" STREQUAL "")
        set(${out_var} "HTTPHEADER" "Authorization: Bearer $ENV{GH_TOKEN}" PARENT_SCOPE)
    endif()
endfunction()

function(gh_resolve_latest out_var)
    set(${out_var} "" PARENT_SCOPE)

    # The latest release may point to a nightly build
    set(tmp "${UI_BINARY_DIR}/nightly-tag.txt")
    file(DOWNLOAD "https://github.com/${GH_REPO}/releases/latest/download/nightly-tag.txt" "${tmp}"
        STATUS status TIMEOUT 30
    )
    list(GET status 0 rc)
    if(rc EQUAL 0)
        file(READ "${tmp}" tag)
        string(STRIP "${tag}" tag)
        set(${out_var} "${tag}" PARENT_SCOPE)
        return()
    endif()

    # Use the GitHub API
    set(tmp "${UI_BINARY_DIR}/latest-release.json")
    gh_auth_headers(auth_headers)

    file(DOWNLOAD "https://api.github.com/repos/${GH_REPO}/releases/latest" "${tmp}"
        STATUS status TIMEOUT 30 ${auth_headers}
    )
    list(GET status 0 rc)
    if(NOT rc EQUAL 0)
        list(GET status 1 errmsg)
        message(STATUS "UI: failed to resolve latest release: ${errmsg}")
        return()
    endif()

    file(READ "${tmp}" json)
    string(JSON tag ERROR_VARIABLE err GET "${json}" tag_name)
    if(NOT "${err}" STREQUAL "NOTFOUND" OR "${tag}" STREQUAL "")
        message(STATUS "UI: no tag_name in latest release response")
        return()
    endif()
    set(${out_var} "${tag}" PARENT_SCOPE)
endfunction()

function(gh_download version out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    set(archive "${UI_BINARY_DIR}/ui.tar.gz")
    set(url "https://github.com/${GH_REPO}/releases/download/${version}/llama-${version}-ui.tar.gz")
    gh_auth_headers(auth_headers)

    message(STATUS "UI: downloading ${url}")

    file(DOWNLOAD "${url}" "${archive}"
        STATUS status TIMEOUT 300 ${auth_headers}
    )
    list(GET status 0 rc)
    if(NOT rc EQUAL 0)
        list(GET status 1 errmsg)
        message(STATUS "UI: download from ${version} failed: ${errmsg}")
        return()
    endif()

    # Release archives nest assets in a llama-<version>/ directory
    set(extract_dir "${UI_BINARY_DIR}/ui-extract")
    file(REMOVE_RECURSE "${extract_dir}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${extract_dir}")

    set(asset_dir "${extract_dir}")
    file(GLOB nested "${extract_dir}/*/index.html")
    get_filename_component(asset_dir "${nested}" DIRECTORY)

    if(NOT EXISTS "${asset_dir}/index.html")
        message(STATUS "UI: archive from ${version} is missing required assets")
        return()
    endif()

    # Clear DIST_DIR to remove stale files first
    file(REMOVE_RECURSE "${DIST_DIR}")
    file(RENAME "${asset_dir}" "${DIST_DIR}")

    message(STATUS "UI: archive extracted")
    set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(gh_provision version out_var)
    set(${out_var} FALSE PARENT_SCOPE)

    set(stamp "")
    if(EXISTS "${STAMP_FILE}")
        file(READ "${STAMP_FILE}" stamp)
        string(STRIP "${stamp}" stamp)
    endif()

    # release assets are immutable per tag, so a matching stamp means up-to-date
    if(EXISTS "${DIST_DIR}/index.html" AND "${stamp}" STREQUAL "${version}")
        message(STATUS "UI: release '${version}' already provisioned, skipping download")
        set(${out_var} TRUE PARENT_SCOPE)
        return()
    endif()

    gh_download("${version}" GH_OK)
    if(GH_OK)
        file(WRITE "${STAMP_FILE}" "${version}")
        message(STATUS "UI: release download succeeded, stamp updated (${version})")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        message(STATUS "UI: release download from '${version}' failed")
    endif()
endfunction()

function(emit_files dist_dir)
    # If gzip is requested, compress every asset into a parallel _gzip/ tree
    # the structure stays the same; for ex: /abc/def --> /_gzip/abc/def
    # embed.cpp will check for _gzip and will pick it up
    if(LLAMA_UI_GZIP AND EXISTS "${dist_dir}/index.html")
        find_program(GZIP_EXECUTABLE gzip)
        if(NOT GZIP_EXECUTABLE)
            message(WARNING "UI: LLAMA_UI_GZIP requested but gzip not found, embedding uncompressed")
        else()
            set(gzip_dir "${dist_dir}/_gzip")
            file(REMOVE_RECURSE "${gzip_dir}")
            file(GLOB_RECURSE all_files RELATIVE "${dist_dir}" "${dist_dir}/*")
            foreach(f ${all_files})
                get_filename_component(dst_dir "${gzip_dir}/${f}" DIRECTORY)
                file(MAKE_DIRECTORY "${dst_dir}")
                execute_process(
                    COMMAND "${GZIP_EXECUTABLE}" -c "${dist_dir}/${f}"
                    OUTPUT_FILE "${gzip_dir}/${f}"
                    RESULT_VARIABLE gz_rc
                )
                if(NOT gz_rc EQUAL 0)
                    message(FATAL_ERROR "UI: gzip failed for ${f}")
                endif()
            endforeach()
            message(STATUS "UI: gzip compression applied (${gzip_dir})")
        endif()
    endif()

    set(args "${UI_CPP}" "${UI_H}")
    if(EXISTS "${dist_dir}/index.html")
        list(APPEND args "${dist_dir}")
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
if(EXISTS "${SRC_DIST_DIR}/index.html")
    message(STATUS "UI: using pre-built assets from ${SRC_DIST_DIR}")
    emit_files("${SRC_DIST_DIR}")
    return()
endif()

# Resolve version from git build-info if not explicitly set
set(provisioned FALSE)
resolve_version(VERSION)
set(UI_VERSION "${VERSION}")

# ---------------------------------------------------------------------------
# 2. Priority 2: GitHub release download of the resolved version (if GH_ENABLED=ON)
# ---------------------------------------------------------------------------
if(GH_ENABLED AND NOT "${VERSION}" STREQUAL "")
    gh_provision("${VERSION}" GH_OK)
    if(GH_OK)
        set(provisioned TRUE)
    endif()
endif()

# ---------------------------------------------------------------------------
# 3. Priority 3: npm build (BUILD_UI=ON) or latest release (BUILD_UI=OFF)
# ---------------------------------------------------------------------------
if(NOT provisioned)
    if(BUILD_UI)
        npm_build(NPM_OK)
        if(NPM_OK)
            set(provisioned TRUE)
        endif()
    elseif(GH_ENABLED)
        gh_resolve_latest(LATEST)
        if(NOT "${LATEST}" STREQUAL "")
            gh_provision("${LATEST}" GH_OK)
            if(GH_OK)
                set(provisioned TRUE)
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 4. Fallback: warn about stale or missing assets, then emit whatever we have
# ---------------------------------------------------------------------------
if(NOT provisioned)
    if(EXISTS "${DIST_DIR}/index.html")
        message(WARNING "UI: provisioning failed; embedding stale assets from ${DIST_DIR}")
    else()
        message(WARNING "UI: no assets available - building without an embedded UI. "
                        "In a disconnected environment, download the pre-built UI "
                        "from a llama.cpp release at "
                        "https://github.com/ggml-org/llama.cpp/releases and "
                        "extract to tools/ui/dist.")
    endif()
endif()

emit_files("${DIST_DIR}")

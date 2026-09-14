# Embed the git HEAD (branch + commit) of this repository into a custom ELF
# section `.self_id` so it can be inspected with:
#   readelf -p .self_id libixformer.so
# Layout (single line): [branch_name] <full commit SHA>
# Branch is `HEAD` when detached.
#
# Uses an always-run target rather than POST_BUILD: an incremental build that
# does not relink the library would otherwise leave a stale commit embedded.
# The stamping script skips objcopy when the value is already current.
function(AddCommitInfo tgt)
    if(NOT TARGET ${tgt})
        message(STATUS "AddCommitInfo: target ${tgt} does not exist, skip")
        return()
    endif()

    find_package(Git QUIET)
    if(NOT CMAKE_OBJCOPY)
        find_program(CMAKE_OBJCOPY NAMES objcopy)
    endif()
    find_program(SELF_ID_READELF NAMES llvm-readelf readelf)

    set(_skip_reason "")
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
        set(_skip_reason "${CMAKE_SOURCE_DIR}/.git not found")
    elseif(NOT Git_FOUND)
        set(_skip_reason "git not found")
    elseif(NOT CMAKE_OBJCOPY)
        set(_skip_reason "objcopy not found")
    endif()

    # The target is created even when stamping is impossible, so build drivers
    # (setup.py builds ${tgt}_self_id explicitly) never fail on a missing target.
    if(_skip_reason)
        message(STATUS "AddCommitInfo: ${_skip_reason}, ${tgt} left unstamped")
        add_custom_target(${tgt}_self_id ALL
            COMMAND ${CMAKE_COMMAND} -E true
            COMMENT "Skipping .self_id for ${tgt} (${_skip_reason})")
        add_dependencies(${tgt}_self_id ${tgt})
        return()
    endif()

    add_custom_target(${tgt}_self_id ALL
        COMMAND ${CMAKE_COMMAND}
                -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                -DSRC_DIR=${CMAKE_SOURCE_DIR}
                -DTARGET_FILE=$<TARGET_FILE:${tgt}>
                -DOBJCOPY_EXECUTABLE=${CMAKE_OBJCOPY}
                -DREADELF_EXECUTABLE=${SELF_ID_READELF}
                -P ${CMAKE_SOURCE_DIR}/cmake/WriteSelfId.cmake
        COMMENT "Embedding git commit and branch into ${tgt} (.self_id)"
        VERBATIM)
    add_dependencies(${tgt}_self_id ${tgt})
endfunction()

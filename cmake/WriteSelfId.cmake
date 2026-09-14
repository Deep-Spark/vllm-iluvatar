# Stamp `[branch] <sha>` into the `.self_id` ELF section of TARGET_FILE.
#
# Invoked via `cmake -P` from AddCommitInfo's always-run target, so the section
# is refreshed even on incremental builds where the library is not relinked.
# objcopy only runs when the embedded value is stale, keeping the no-op case
# cheap for large libraries.
#
# Expects: GIT_EXECUTABLE, SRC_DIR, TARGET_FILE, OBJCOPY_EXECUTABLE
#          READELF_EXECUTABLE (optional; enables the up-to-date check)

if(NOT EXISTS "${TARGET_FILE}")
  message(STATUS "WriteSelfId: ${TARGET_FILE} not found, skip")
  return()
endif()

execute_process(
  COMMAND ${GIT_EXECUTABLE} -c safe.directory=${SRC_DIR} -C ${SRC_DIR} rev-parse --abbrev-ref HEAD
  OUTPUT_VARIABLE _branch
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rc1)
execute_process(
  COMMAND ${GIT_EXECUTABLE} -c safe.directory=${SRC_DIR} -C ${SRC_DIR} rev-parse HEAD
  OUTPUT_VARIABLE _sha
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rc2)
if(NOT _rc1 EQUAL 0 OR NOT _rc2 EQUAL 0)
  message(FATAL_ERROR "WriteSelfId: failed to read git HEAD/branch in ${SRC_DIR}")
endif()

set(_self_id "[${_branch}] ${_sha}")

# Skip the rewrite when the library already carries this exact value.
if(READELF_EXECUTABLE)
  execute_process(
    COMMAND ${READELF_EXECUTABLE} -p .self_id ${TARGET_FILE}
    OUTPUT_VARIABLE _dump
    ERROR_QUIET)
  string(FIND "${_dump}" "${_self_id}" _found)
  if(NOT _found EQUAL -1)
    return()
  endif()
endif()

set(_txt "${TARGET_FILE}.self_id.txt")
file(WRITE "${_txt}" "${_self_id}\n")
execute_process(COMMAND ${OBJCOPY_EXECUTABLE} --remove-section=.self_id ${TARGET_FILE})
execute_process(
  COMMAND ${OBJCOPY_EXECUTABLE} --add-section .self_id=${_txt} ${TARGET_FILE}
  RESULT_VARIABLE _rc3)
file(REMOVE "${_txt}")
if(NOT _rc3 EQUAL 0)
  message(FATAL_ERROR "WriteSelfId: objcopy failed on ${TARGET_FILE}")
endif()
message(STATUS "WriteSelfId: ${TARGET_FILE} <- ${_self_id}")

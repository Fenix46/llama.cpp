include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    paged-request.h
    server-tools.cpp
    server-tools.h
    scheduler/request_state.cpp
    scheduler/request_state.h
    scheduler/request_group.cpp
    scheduler/request_group.h
    scheduler/request_lifecycle.cpp
    scheduler/request_lifecycle.h
    scheduler/sched_policy.cpp
    scheduler/sched_policy.h
    scheduler/prefill_policy.cpp
    scheduler/prefill_policy.h
    scheduler/admission_controller.cpp
    scheduler/admission_controller.h
    scheduler/reservation_model.cpp
    scheduler/reservation_model.h
    scheduler/block_manager.cpp
    scheduler/block_manager.h
    scheduler/prefix_reuse_manager.cpp
    scheduler/prefix_reuse_manager.h
    scheduler/lineage_manager.h
    scheduler/batch_plan.cpp
    scheduler/batch_plan.h
    scheduler/batch_planner.cpp
    scheduler/batch_planner.h
    scheduler/step_executor.cpp
    scheduler/step_executor.h
    scheduler/sampling_executor.cpp
    scheduler/sampling_executor.h
    scheduler/speculative_executor.cpp
    scheduler/speculative_executor.h
    scheduler/paged_scheduler.cpp
    scheduler/paged_scheduler.h
    scheduler/scheduler_metrics.cpp
    scheduler/scheduler_metrics.h
    scheduler/scheduler_core.cpp
    scheduler/scheduler_core.h
    scheduler/legacy_slot_adapter.cpp
    scheduler/legacy_slot_adapter.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

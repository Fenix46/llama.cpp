#pragma once

#include "../server-task.h"

#include <functional>
#include <utility>

namespace server_scheduler {

// Minimal internal boundary between the server/API layer (server_context_impl)
// and the runtime scheduler. This is the Fase 1 interface from the paged
// scheduler migration plan: it confines the launch/cancel/tick entry points
// behind a backend so server_context_impl no longer dispatches paged-vs-legacy
// inline for the hot loop.
//
// The interface is deliberately small. metrics/save/restore/erase are NOT part
// of it yet because their semantics in paged mode are still being normalized
// (see plan Fase 6). The concrete backends are thin wrappers that delegate to
// the existing server_context_impl methods via callbacks, so introducing them
// does not move state or change behavior.
class ServerSchedulerBackend {
public:
    virtual ~ServerSchedulerBackend() = default;

    // Dispatch a COMPLETION/INFILL/EMBEDDING/RERANK task to the runtime.
    // The backend is responsible for admission/defer and launch.
    virtual void launch_completion(server_task && task) = 0;

    // Cancel an in-flight task by its target task id.
    virtual void cancel(int id_target) = 0;

    // Advance the runtime one scheduler iteration.
    virtual void tick() = 0;
};

// Thin wrapper backend that delegates each entry point to server_context_impl
// callbacks. Both the legacy slot path and the paged path use this type during
// Fase 1; the only difference is which server_context_impl methods the
// callbacks bind to. Later phases can replace these wrappers with backends that
// own their runtime state directly.
class CallbackSchedulerBackend final : public ServerSchedulerBackend {
public:
    struct Callbacks {
        std::function<void(server_task &&)> launch_completion;
        std::function<void(int)>            cancel;
        std::function<void()>               tick;
    };

    explicit CallbackSchedulerBackend(Callbacks cbs) : cbs_(std::move(cbs)) {}

    void launch_completion(server_task && task) override { cbs_.launch_completion(std::move(task)); }
    void cancel(int id_target) override { cbs_.cancel(id_target); }
    void tick() override { cbs_.tick(); }

private:
    Callbacks cbs_;
};

} // namespace server_scheduler

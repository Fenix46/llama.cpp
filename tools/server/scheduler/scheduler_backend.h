#pragma once

#include "../server-task.h"

#include <functional>
#include <utility>

// Forward declaration: the paged backend borrows (does not own) the server
// context. Its method bodies are defined in server-context.cpp where the full
// definition is visible, so the header stays decoupled from it.
struct server_context_impl;

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

    // Handle a slot action (save/restore/erase). Returns true if the backend
    // fully handled it (result/error already sent) and the caller must NOT run
    // the legacy slot path. Base default: not handled — preserves the legacy
    // slot behavior, which never routed through a backend.
    virtual bool handle_slot_action(server_task & task) { (void) task; return false; }

    // Populate the /metrics slots view (slots_data + idle/processing counts).
    // Returns true if the backend produced the data; base default returns false
    // so the caller runs the legacy per-slot iteration unchanged.
    virtual bool collect_slots_metrics(json & slots_data, int & n_idle, int & n_processing) {
        (void) slots_data; (void) n_idle; (void) n_processing; return false;
    }

    // Append backend-specific fields to the /metrics prefix_cache_data object
    // (e.g. paged seq_leases). Base default: no-op (legacy adds nothing).
    virtual void augment_prefix_cache_metrics(json & prefix_cache_data) { (void) prefix_cache_data; }

    // Whether the /slots endpoint and the endpoint_slots prop are meaningful.
    // Base default: true (legacy has server_slot objects). Paged has no slots.
    virtual bool supports_slots_endpoint() const { return true; }
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

// Concrete backend for the paged scheduler. Unlike CallbackSchedulerBackend it
// is paged-specific: it borrows a server_context_impl pointer and forwards the
// entry points to the paged runtime methods. State (paged_requests, leases,
// lifecycle, ...) stays owned by server_context_impl; this backend only routes.
// Method bodies live in server-context.cpp (full server_context_impl visible).
class PagedSchedulerBackend final : public ServerSchedulerBackend {
public:
    explicit PagedSchedulerBackend(::server_context_impl * ctx) : ctx_(ctx) {}

    void launch_completion(server_task && task) override;
    void cancel(int id_target) override;
    void tick() override;
    bool handle_slot_action(server_task & task) override;
    bool collect_slots_metrics(json & slots_data, int & n_idle, int & n_processing) override;
    void augment_prefix_cache_metrics(json & prefix_cache_data) override;
    bool supports_slots_endpoint() const override { return false; }

private:
    ::server_context_impl * ctx_;
};

} // namespace server_scheduler

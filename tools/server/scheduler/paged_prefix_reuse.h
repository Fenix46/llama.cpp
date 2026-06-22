#pragma once

#include "prefix_reuse_manager.h"
#include "request_state.h"

#include "common.h"

#include <cstdint>
#include <functional>
#include <string>

namespace server_scheduler {

struct PagedPrefixReuseConfig {
    common_params * params_base = nullptr;
    PrefixReuseManager * prefix_cache = nullptr;
    std::string model_name;

    bool paged_block_prefix_supported = true;
    bool paged_enable_block_prefix_cache = true;
    bool paged_enable_same_seq_append = false;
    bool paged_enable_donor_seq_fallback = false;

    std::function<RequestState *(int32_t)> find_request_by_seq_id;
};

class PagedPrefixReuse {
public:
    explicit PagedPrefixReuse(PagedPrefixReuseConfig config);

    PrefixReuseMetadata build_metadata(const RequestState & req) const;
    void execute_plan(RequestState & req) const;

private:
    PagedPrefixReuseConfig config_;

    static bool can_use_prefix_copy(const RequestState & req);
};

} // namespace server_scheduler

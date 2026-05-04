#pragma once

#include <cstdint>
#include <vector>

namespace server_scheduler {

struct RequestGroup {
    int32_t parent_request_id = -1;
    std::vector<int32_t> child_request_ids;

    bool valid() const {
        return parent_request_id >= 0;
    }
};

} // namespace server_scheduler

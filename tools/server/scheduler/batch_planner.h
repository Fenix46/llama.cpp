#pragma once

#include "batch_plan.h"

namespace server_scheduler {

class BatchPlanner {
public:
    BatchPlan build() const { return {}; }
};

} // namespace server_scheduler

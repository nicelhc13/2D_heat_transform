import "regent"

local c = regentlib.c
local fm = require("std/format")

local prof_constants = terralib.includec("heat_transfer/profiling/profiling_constants.h")

local bridgemap = {}

-- Timer tasks for demarcation.
-- These tasks are empty at application level, but
-- when these are mapped, the runtime turns on/off the
-- corresponding timers.
-- The time results would be used to evaluate quality
-- of a mapping result.
-- It exploits event system of the Legion, which is
-- similar with CUDA events.
-- Starting/stopping timers register and waits for events during
-- set_tunable_value() until map_task() is done.
-- The event is triggered at the end of the report_profiling() when
-- all timer handling is done, mapping is done, and the runtime
-- gets all timer reulsts. 

task start_timer()
  return
end

task stop_timer()
  return
end

task output_mapping()
  return
end

__demand(__inline)
task bridgemap.begin_profile()
  -- First, put a fence across all tasks/processors.
  c.legion_runtime_issue_execution_fence(__runtime(), __context())
  start_timer()
  -- After the start timer processing is done (at map_task()), destroy
  -- the event.
  var f = c.legion_runtime_select_tunable_value(__runtime(), __context(),
              prof_constants.TUNABLE_VALUE_START, 0, 0)
  c.legion_future_get_untyped_pointer(f)
  c.legion_future_destroy(f)
end

__demand(__inline)
task bridgemap.end_profile()
  c.legion_runtime_issue_execution_fence(__runtime(), __context())
  stop_timer()
  var f = c.legion_runtime_select_tunable_value(__runtime(), __context(),
              prof_constants.TUNABLE_VALUE_STOP, 0, 0)
  c.legion_future_get_untyped_pointer(f)
  c.legion_future_destroy(f)
end

__demand(__inline)
task bridgemap.output_best_mapping()
  c.legion_runtime_issue_execution_fence(__runtime(), __context())
  output_mapping()
end

return bridgemap

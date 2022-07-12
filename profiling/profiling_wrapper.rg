import "regent"

local c = regentlib.c
local fm = require("std/format")

local prof_constants = terralib.includec("heat_transfer/profiling/profiling_constants.h")

local bridgemap = {}

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
  c.legion_runtime_issue_execution_fence(__runtime(), __context())
  start_timer()
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

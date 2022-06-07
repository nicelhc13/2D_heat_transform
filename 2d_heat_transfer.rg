import "regent"

local c = regentlib.c
local fm = require("std/format")  

task main()
  var plate_len = 50
  var max_iter_time = 750

  -- Constants.
  var alpha = 2
  var delta_x = 1
  var delta_t = (float(delta_x) * delta_x) / (4 * alpha)
  var gamma = (alpha * delta_t) / (delta_x * delta_x)

  fm.println("Alpha {} Delta X {} Delta T {.3} Gamma {.3} ",
      alpha, delta_x, delta_t, gamma)
end

regentlib.start(main)

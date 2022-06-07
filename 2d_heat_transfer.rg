import "regent"

local c = regentlib.c
local fm = require("std/format")  

--task make_plate_partition(plate : region(ispace(int3d), float),
--                          blocks : ispace(int3d),
--                          n : int64, nt : int64)
--  for bi in blocks do
--  end
--end

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

  var time_stacked_plate_space =
      ispace(int3d, {x = max_iter_time, y = plate_len, z = plate_len})
  var time_stakced_plates = region(time_stacked_plate_space, float)

  var plate_space = ispace(int3d, {x = 1, y = plate_len, z = plate_len})
  var coloring = c.legion_domain_point_coloring_create()
  var plates = partition(disjoint, time_stakced_plates, coloring, plate_space)

  for time = 0, max_iter_time do
    for i in plate_space[int3d { x = time }] do
      fm.print(" Time: {}, [{}, {}] = {} \n",
          time, i.y, i.z, plates[int3d{x = time, y = i.y, z = i.z}])
    end
  end

--  for time = 0, max_iter_time do
--    for i in plate_space[time] do
--      fm.print(" Time: {}, [{}, {}] = {} \n",
--          time, i.y, i.z, plates[i])
--    end
--  end

--  for p in plate_space do
--    fm.print(" [{}, {}, {}] = {} \n", p.x, p.y, p.z, plate[p])
--  end

  var plate_top = 100.0
  var plate_left = 0.0
  var plate_right = 0.0
  var plate_bottom = 100.0

  -- var blocks = make_plate_partition(plate, 

end

regentlib.start(main)

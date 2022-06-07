import "regent"

local c = regentlib.c
local fm = require("std/format")  

task make_plate_partition(plate : region(ispace(int2d), float),
                          n : int64, block_len : int64)
  var coloring = c.legion_domain_point_coloring_create()
  var lo = int2d { x = 0, y = 0 }
  var hi = int2d { x = block_len - 1, y = block_len - 1 }
  var idx = int2d { x = 0, y = 0 }
  var block_dim = int2d { x = 0, y = 0 }
  while true do
    if hi.x > n or hi.y > n then
      break
    end
    while true do
      if hi.x > n or hi.y > n then
        break
      end
      var rect = rect2d { lo = lo, hi = hi }
      -- fm.println("{}, {}th block: [{}, {}] to [{}, {}]", idx.x, idx.y,
      --           lo.x, lo.y, hi.x, hi.y)
      c.legion_domain_point_coloring_color_domain(coloring, idx, rect)
      lo = int2d { x = lo.x, y = lo.y + 2}
      hi = int2d { x = hi.x, y = hi.y + 2}
      idx = int2d { x = idx.x, y = idx.y + 1}
      block_dim.y += 1
    end
    lo = int2d { x = lo.x + 2, y = 0 }
    hi = int2d { x = hi.x + 2, y = 0 }
    idx = int2d { x = idx.x + 1, y = 0}
    block_dim.x += 1
  end
  var is_block = ispace(int2d, {x = block_dim.x, y = block_dim.y })
  var p = partition(aliased, plate, coloring, is_block)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

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

  var is_plate =
      ispace(int2d, {x = plate_len, y = plate_len})
  var old_plate = region(is_plate, float)
  var new_plate = region(is_plate, float)

  var block_len = 3
  var private_blocks = ispace(int2d, {x = block_len - 1, y = block_len - 1})

  var old_plate_tiles = make_plate_partition(old_plate, plate_len, block_len)
  var new_plate_tiles = make_plate_partition(new_plate, plate_len, block_len)

--  for time = 0, max_iter_time do
--    for i in plate_space[int3d { x = time }] do
--      fm.print(" Time: {}, [{}, {}] = {} \n",
--          time, i.y, i.z, plates[int3d{x = time, y = i.y, z = i.z}])
--    end
--  end

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

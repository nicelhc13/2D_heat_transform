import "regent"

local c = regentlib.c
local fm = require("std/format")  

task make_plate_partition(plate : region(ispace(int2d), float),
                          n : int64, block_len : int64)
where reads writes(plate) do
  var coloring = c.legion_domain_point_coloring_create()
  var lo = int2d { x = 0, y = 0 }
  var hi = int2d { x = block_len - 1, y = block_len - 1 }
  var idx = int2d { x = 0, y = 0 }
  var block_dim = 0
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
      --          lo.x, lo.y, hi.x, hi.y)
      c.legion_domain_point_coloring_color_domain(coloring, idx, rect)
      lo = int2d { x = lo.x, y = lo.y + 2}
      hi = int2d { x = hi.x, y = hi.y + 2}
      idx = int2d { x = idx.x, y = idx.y + 1}
    end
    lo = int2d { x = lo.x + 2, y = 0 }
    hi = int2d { x = hi.x + 2, y = block_len - 1 }
    idx = int2d { x = idx.x + 1, y = 0}
    block_dim += 1
  end
  var is_block = ispace(int2d, {x = block_dim, y = block_dim })
  var p = partition(aliased, plate, coloring, is_block)
  c.legion_domain_point_coloring_destroy(coloring)
  -- fm.println("block {} x {}", block_dim, block_dim)
  return p
--  return { prt = p, dim = is_block }
end

task initialize_tiles(plate : region(ispace(int2d), float),
                      plate_len : float, plate_top : float,
                      plate_left : float, plate_right : float,
                      plate_bottom : float)
where reads writes(plate) do
  for p in plate do
    if p.x >= (plate_len - 1) then
      plate[p] = plate_top
    end
    if p.y < 1 then
      plate[p] = plate_left
    end
    if p.x < 1 and p.y >= 1 then
      plate[p] = plate_bottom
    end
    if p.y >= (plate_len - 1) then
      plate[p] = plate_right
    end
  end
end

task sweep(curr_plate : region(ispace(int2d), float),
           next_plate : region(ispace(int2d), float),
           plate_len : int64, gamma : float)
where reads(curr_plate), reads writes(next_plate) do
  for i in curr_plate do
    if (i.x ~= 0 and i.x ~= plate_len - 1) and
       (i.y ~= 0 and i.y ~= plate_len - 1) then
      var is_i0 = int2d { x = i.x + 1, y = i.y }
      var is_i1 = int2d { x = i.x - 1, y = i.y }
      var is_i2 = int2d { x = i.x, y = i.y + 1 }
      var is_i3 = int2d { x = i.x, y = i.y - 1 }
      -- fm.println(" {}, {}, 1 {}, 2 {}, 3{}, 4{}, 5 {} ",
      --    i.x, i.y, curr_plate[is_i0], curr_plate[is_i1],
      --    curr_plate[is_i2], curr_plate[is_i3], curr_plate[i])
      next_plate[i] = gamma * (curr_plate[is_i0] + curr_plate[is_i1] +
         curr_plate[is_i2] + curr_plate[is_i3] - 4 * curr_plate[i]) +
         curr_plate[i]
    end
  end
end

task main()
  var plate_len = 50
  var max_iter_time = 100

  -- Constants.
  var alpha = 2
  var delta_x = 1
  var delta_t = (float(delta_x) * delta_x) / (4 * alpha)
  var gamma = (alpha * delta_t) / (delta_x * delta_x)

  fm.println("Alpha {} Delta X {} Delta T {.3} Gamma {.3} ",
      alpha, delta_x, delta_t, gamma)

  var is_plate =
      ispace(int2d, {x = plate_len, y = plate_len})
  var curr_plate = region(is_plate, float)
  var next_plate = region(is_plate, float)

  -- Initialize regions first.
  fill (curr_plate, 0)
  fill (next_plate, 0)

  var block_len = 3
  var num_blocks = (plate_len - block_len) / 2 + 1
  if ((plate_len - block_len) % 2 > 0) then
    num_blocks += 1
  end
  var is_blocks = ispace(int2d, { num_blocks, num_blocks })


--  var { curr_plate_tiles = prt, is_curr_plate_tiles = dim } =
--              make_plate_partition(curr_plate, plate_len, block_len)
--  var { next_plate_tiles = prt, is_next_plate_tiles = dim } =
--             make_plate_partition(next_plate, plate_len, block_len)
  var curr_plate_tiles = make_plate_partition(curr_plate, plate_len, block_len)
  var next_plate_tiles = make_plate_partition(next_plate, plate_len, block_len)


  var plate_top = 100.0
  var plate_left = 0.0
  var plate_right = 0.0
  var plate_bottom = 0.0

  for p in is_blocks do
    initialize_tiles(next_plate_tiles[p], plate_len, plate_top, plate_left,
                     plate_right, plate_bottom)
    initialize_tiles(curr_plate_tiles[p], plate_len, plate_top, plate_left,
                     plate_right, plate_bottom)
  end

  for time = 0, max_iter_time do
    if time % 2 == 0 then
      -- __demand(__index_launch): XXX(lhc): this is not possible due to
      -- overlapped regions. Q: still there would be parallelism, but
      -- does Legion really parallelize it?
      for p in is_blocks do
        sweep(curr_plate_tiles[p], next_plate_tiles[p], plate_len, gamma)
      end
    else
      -- __demand(__index_launch)
      for p in is_blocks do
        sweep(next_plate_tiles[p], curr_plate_tiles[p], plate_len, gamma)
      end
    end
  end

  for p in next_plate do
    -- fm.println(" Tile [{} {}] ", p.x, p.y)
    -- for i in curr_plate_tiles[p] do
    fm.println("[{} {}] {} ", p.x, p.y, next_plate[p])
    -- end
  end

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

  -- var blocks = make_plate_partition(plate, 

end

regentlib.start(main)

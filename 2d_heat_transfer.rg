import "regent"

local c = regentlib.c
local fm = require("std/format")  

task make_read_plate_partition(plate : region(ispace(int2d), float),
                          n : int64, block_len : int64)
where reads(plate) do
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
      lo = int2d { x = lo.x, y = lo.y + 1}
      hi = int2d { x = hi.x, y = hi.y + 1}
      idx = int2d { x = idx.x, y = idx.y + 1}
    end
    lo = int2d { x = lo.x + 1, y = 0 }
    hi = int2d { x = hi.x + 1, y = block_len - 1 }
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

task make_write_plate_partition(plate : region(ispace(int2d), float), n : int64)
where reads(plate) do
  return partition(equal, plate, ispace(int2d, {x = n, y = n}))
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

task sweep(plate1 : region(ispace(int2d), float),
           plate2 : region(ispace(int2d), float),
           plate_len : int64, gamma : float)
where reads(plate1), reads writes(plate2) do
  for i in plate1 do
    if (i.x ~= 0 and i.x ~= plate_len - 1) and
       (i.y ~= 0 and i.y ~= plate_len - 1) then
      var is_i0 = int2d { x = i.x + 1, y = i.y }
      var is_i1 = int2d { x = i.x - 1, y = i.y }
      var is_i2 = int2d { x = i.x, y = i.y + 1 }
      var is_i3 = int2d { x = i.x, y = i.y - 1 }
      -- fm.println(" {}, {}, 1 {}, 2 {}, 3{}, 4{}, 5 {} ",
      --    i.x, i.y, plate1[is_i0], plate1[is_i1],
      --    plate1[is_i2], plate1[is_i3], plate1[i])
      plate2[i] = gamma * (plate1[is_i0] + plate1[is_i1] +
         plate1[is_i2] + plate1[is_i3] - 4 * plate1[i]) +
         plate1[i]
    end
  end
end

task sweep2(cp : region(ispace(int2d), float),
            np : region(ispace(int2d), float),
            plate_len : int64, gamma : float)
where reads(cp), reads writes(np) do
  var low = cp.bounds.lo
  var central = int2d{ x = low.x + 1, y = low.y + 1 }
  var is_i0 = int2d { x = central.x + 1, y = central.y }
  var is_i1 = int2d { x = central.x - 1, y = central.y }
  var is_i2 = int2d { x = central.x, y = central.y + 1 }
  var is_i3 = int2d { x = central.x, y = central.y - 1 }

  for n in np do
    np[n] = gamma * (cp[is_i0] + cp[is_i1] + cp[is_i2] +
                cp[is_i3] - 4 * cp[central]) + cp[central]
    -- fm.println(" {} {}, 1 {} 2 {} 3 {} 4 {} 5 {}",
    --             n.x, n.y, cp[is_i0], cp[is_i1],
    --            cp[is_i2], cp[is_i3], np[n])
  end
  -- var i = 0
  -- for c in curr_plate do
    -- fm.println("{} {} {}", i, c.x, c.y)
    -- fm.println("{} {} {}", i, curr_plate.bounds.lo.x, curr_plate.bounds.lo.y)
    --fm.println("{} {} {}", i, curr_plate.bounds.hi.x, curr_plate.bounds.hi.y)
    -- i += 1
  -- end
  -- fm.println("\n\n")
end

task sweep3(cp : region(ispace(int2d), float),
            np : region(ispace(int2d), float),
            plate_len : int64, gamma : float)
where reads(cp), reads writes(np) do
  for n in np do
      --   if (cord_to_update.x ~= 0 and cord_to_update.x ~= plate_len - 1) and
      --         (cord_to_update.y ~= 0 and cord_to_update.y ~= plate_len - 1) then

    if (n.x ~= 0 and n.x ~= plate_len - 1 and n.y ~= 0 and n.y ~= plate_len - 1) then
      var is_i0 = int2d { x = n.x + 1, y = n.y }
      var is_i1 = int2d { x = n.x - 1, y = n.y }
      var is_i2 = int2d { x = n.x, y = n.y + 1 }
      var is_i3 = int2d { x = n.x, y = n.y - 1 }
      np[n] = gamma * (cp[is_i0] + cp[is_i1] + cp[is_i2] +
                  cp[is_i3] - 4 * cp[n]) + cp[n]
      -- fm.println("{}, {} = {}, 1 {} 2 {} 3 {} 4 {}", n.x, n.y, np[n], cp[is_i0],
      --           cp[is_i1], cp[is_i2], cp[is_i3])
    end
  end
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
  var plate1 = region(is_plate, float)
  var plate2 = region(is_plate, float)

  -- Initialize regions first.
  fill (plate1, 0)
  fill (plate2, 0)

  var block_len = 3
  -- var num_blocks = (plate_len - 2)
  var num_blocks = plate_len / block_len
  if plate_len % block_len > 1 then
    num_blocks += 1
  end
  fm.println("Blocks {}", num_blocks)
  var is_blocks = ispace(int2d, { num_blocks, num_blocks })


  -- var { plate1_tiles = prt, is_plate1_tiles = dim } =
  --            make_read_plate_partition(plate1, plate_len, block_len)
  -- var { plate2_tiles = prt, is_plate2_tiles = dim } =
  --            make_read_plate_partition(plate2, plate_len, block_len)
  var plate1_tiles = make_read_plate_partition(plate1, plate_len, block_len)
  var plate2_tiles = make_read_plate_partition(plate2, plate_len, block_len)

  var next_plate1_tiles = make_write_plate_partition(plate1, num_blocks) 
  var next_plate2_tiles = make_write_plate_partition(plate2, num_blocks)

  --  for p in is_blocks do
  --    for x in next_plate1_tiles[p] do
  --      fm.println("[ {}, {} ]", x.x, x.y)
  --    end
  --    fm.println("\n")
  --  end

  var plate_top = 100.0
  var plate_left = 0.0
  var plate_right = 0.0
  var plate_bottom = 0.0

  initialize_tiles(plate2, plate_len, plate_top, plate_left,
                   plate_right, plate_bottom)
  initialize_tiles(plate1, plate_len, plate_top, plate_left,
                   plate_right, plate_bottom)

  fm.println("Start computation.")
  for time = 0, max_iter_time do
    fm.println("Time: {}", time)
    if time % 2 == 0 then
      __demand(__index_launch)
      for p in is_blocks do
        sweep3(plate1, next_plate2_tiles[p], plate_len, gamma)
      end
      -- __demand(__index_launch): XXX(lhc): this is not possible due to
      -- overlapped regions. Q: still there would be parallelism, but
      -- does Legion really parallelize it?
      -- for p in is_blocks do
      --   var cord_to_update = int2d { x = plate1_tiles[p].bounds.lo.x + 1,
      --                        y = plate1_tiles[p].bounds.lo.y + 1 }
      --   if (cord_to_update.x ~= 0 and cord_to_update.x ~= plate_len - 1) and
      --         (cord_to_update.y ~= 0 and cord_to_update.y ~= plate_len - 1) then
      --     sweep2(plate1_tiles[p], next_plate2_tiles[cord_to_update], plate_len, gamma)
      --   end
      -- end
    else
      __demand(__index_launch)
      for p in is_blocks do
        sweep3(plate2, next_plate1_tiles[p], plate_len, gamma)
      end
      -- for p in is_blocks do
      --   var cord_to_update = int2d { x = plate2_tiles[p].bounds.lo.x + 1,
      --                        y = plate2_tiles[p].bounds.lo.y + 1 }
      --   if (cord_to_update.x ~= 0 and cord_to_update.x ~= plate_len - 1) and
      --         (cord_to_update.y ~= 0 and cord_to_update.y ~= plate_len - 1) then
      --     sweep2(plate2_tiles[p], next_plate1_tiles[cord_to_update], plate_len, gamma)
      --   end
      -- end
    end
  end
  fm.println("Computation completes.")

  for b in ispace(int2d, { x = num_blocks, y = num_blocks }) do
    if max_iter_time % 2 == 0 then
      for i in next_plate1_tiles[b] do
        fm.println("[{} {}] {} ", i.x, i.y, next_plate1_tiles[b][i])
      end
    else
      for i in next_plate2_tiles[b] do
        fm.println("[{} {}] {} ", i.x, i.y, next_plate2_tiles[b][i])
      end
    end
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

  -- var blocks = make_read_plate_partition(plate, 

end

regentlib.start(main)

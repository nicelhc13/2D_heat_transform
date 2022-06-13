import "regent"

local c = regentlib.c
local fm = require("std/format")  

do
  local root_dir = arg[0]:match(".*/") or "./"

  local include_path = ""
  -- Allocate new terra list object.
  local include_dirs = terralib.newlist()
  -- Add all possible include files, to access the mapper object.
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local mapper_cc = root_dir .. "2d_heat_transfer_mapper.cc"
  -- Construct 
  mapper_so = os.tmpname() .. ".so"
  print("Mapper:", mapper_so)
  print("arg 0: ", arg[0], " and root dir: ", root_dir)
  print("include directories: ", include_dirs)

  local cxx = os.getenv("CXX") or "c++"
  local cxx_flags = os.getenv("CXXFLAGS") or ""
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror -fPIC -shared"

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile ".. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("2d_heat_transfer_mapper.h", include_dirs)
end

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
  return p
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

task sweep(cp : region(ispace(int2d), float),
           np : region(ispace(int2d), float),
           plate_len : int64, gamma : float)
where reads(cp), reads writes(np) do
  for n in np do
    if (n.x ~= 0 and n.x ~= plate_len - 1 and n.y ~= 0 and n.y ~= plate_len - 1) then
      var is_i0 = int2d { x = n.x + 1, y = n.y }
      var is_i1 = int2d { x = n.x - 1, y = n.y }
      var is_i2 = int2d { x = n.x, y = n.y + 1 }
      var is_i3 = int2d { x = n.x, y = n.y - 1 }
      np[n] = gamma * (cp[is_i0] + cp[is_i1] + cp[is_i2] +
                  cp[is_i3] - 4 * cp[n]) + cp[n]
    end
  end
end

task main()
  var plate_len = 50
  var max_iter_time = 10

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

  var plate1_tiles = make_read_plate_partition(plate1, plate_len, block_len)
  var plate2_tiles = make_read_plate_partition(plate2, plate_len, block_len)

  var next_plate1_tiles = make_write_plate_partition(plate1, num_blocks) 
  var next_plate2_tiles = make_write_plate_partition(plate2, num_blocks)

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
        sweep(plate1, next_plate2_tiles[p], plate_len, gamma)
      end
    else
      __demand(__index_launch)
      for p in is_blocks do
        sweep(plate2, next_plate1_tiles[p], plate_len, gamma)
      end
    end
  end
  fm.println("Computation completes.")

--  for b in ispace(int2d, { x = num_blocks, y = num_blocks }) do
--    if max_iter_time % 2 == 0 then
--      for i in next_plate1_tiles[b] do
--        fm.println("[{} {}] {} ", i.x, i.y, next_plate1_tiles[b][i])
--      end
--    else
--      for i in next_plate2_tiles[b] do
--        fm.println("[{} {}] {} ", i.x, i.y, next_plate2_tiles[b][i])
--      end
--    end
--  end
end

regentlib.start(main, cmapper.register_mappers)

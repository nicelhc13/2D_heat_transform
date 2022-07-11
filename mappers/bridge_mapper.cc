#include "bridge_mapper.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

// This class is a bridge mapper between application and a RL module.
class BridgeMapper : public DefaultMapper {
public:
  BridgeMapper(MapperRuntime* rt, Machine machine, Processor local_proc,
               const char* mapper_name, std::vector<Processor>* procs_list);

  virtual void select_task_options(const MapperContext ctx,
                                   const Task& task, TaskOptions& output);

  virtual void map_task(const MapperContext ctx, const Task& task,
                        const MapTaskInput& input, MapTaskOutput& output);

  void generate_mapping_using_coord(const MapperContext ctx,
                                    const Task& task,
                                    const MapTaskInput& input,
                                    MapTaskOutput& output);

protected:
  struct Mapping {
  public:
    std::string task_name_;
    Processor::Kind proc_kind_;
    std::vector<Processor> procs_;
    // Processor -> <logical region id, a list of memory for region partitions>.
    std::map<Processor::Kind, std::map<size_t, std::vector<Memory::Kind>>>
        memories_;
  };

private:
  size_t num_iter_;
  // Task ID to mapping.
  std::map<TaskID, Mapping> sampled_mapping_;
  // Cached mapping.
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping> > cached_task_mappings_;
  // GPU processors.
  // Logical index will be used. TODO(hc): is it corresponding to CUDA number?
  std::vector<Processor> GPU_procs_;
};

BridgeMapper::BridgeMapper(MapperRuntime* rt, Machine machine,
                           Processor local_proc, const char* mapper_name,
                           std::vector<Processor>* procs_list)
  : DefaultMapper(rt, machine, local_proc, mapper_name), num_iter_(0) {
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);
  
  for (std::set<Processor>::const_iterator it = all_procs.begin();
       it != all_procs.end(); ++it) {
    switch(it->kind()) {
      case Processor::TOC_PROC:
        {
          printf(" GPU processor ID " IDFMT " \n", it->id);
          GPU_procs_.emplace_back(*it);
          break;
        }
      default:
        break;
    }
  }
  
}

void BridgeMapper::select_task_options(const MapperContext ctx,
                                       const Task& task, TaskOptions& output) {
  printf("Select task options\n");
  std::cout << "Task init procs:" << task.target_proc<< ", no proc:"
    << Processor::NO_PROC << "\n" << std::flush;
  // Eanble the runtime to track, and get valid instances and get premapped
  // region information.
  output.valid_instances = true;
}

static uint32_t calc_gpu_id(DomainPoint dp, Domain dm) {
  // Simplistic policy: ([0:N], [0:B-1]) ([0:N], [B:N]) parititoning.
  uint32_t block_size = dm.hi().point_data[0] / 2;
  uint32_t gpu_id = 0;
  if (dp.point_data[1] > block_size) {
    gpu_id = 1;
  }
  std::cout << dp.point_data[0] << "," << dp.point_data[1] << 
    " on " << dm.lo() << "-" << dm.hi() << " is assigned to GPU" << gpu_id <<
    "\n" << std::flush;
  return gpu_id;
}

void BridgeMapper::map_task(const MapperContext ctx, const Task& task,
                            const MapTaskInput& input, MapTaskOutput& output) {
  std::cout << "Task mapping: " << task.get_task_name() << ", # of regions:" << task.regions.size() << "\n" << std::flush;
  std::cout << ", task domain point:" << task.index_point << ", target proc:" << task.target_proc << "\n" << std::flush;
  //DefaultMapper::map_task(ctx, task, input, output);

  if (num_iter_ == 0 || strstr(task.get_task_name(), "sweep") == NULL) {
    std::cout << "Default mapper is used.\n" << std::flush;
    // The first iteration uses the default mapper.
    DefaultMapper::map_task(ctx, task, input, output);
  } else {
    std::cout << "Sampled mapping is used.\n" << std::flush;
    generate_mapping_using_coord(ctx, task, input, output);
  }

  num_iter_ += 1;
  std::cout << " Target proc:" << output.target_procs[0].kind() << "\n" << std::flush;
}

/// This function maps a task based on its domain point.
/// This is not practical mapping, but is written as a practice to prepare
/// a mapping from outer module. For example, our ultimate goal is to
/// choose the target processor and target memory on external modules (like
/// RL trainer), and then use or create the instances with the equivalent
/// optimizations used in the default mapper.
/// Note that this mapper does not consider REDUCE privilege and INNER TASK.
/// (TODO(hc): I still don't know what the inner task is)
void BridgeMapper::generate_mapping_using_coord(const MapperContext ctx,
                                                const Task& task,
                                                const MapTaskInput& input,
                                                MapTaskOutput& output) {
  //----------------------------------------------------------------------------
  // Choose default option.
  //----------------------------------------------------------------------------

  uint32_t target_gpu_id = calc_gpu_id(task.index_point, task.index_domain); 
  /* Intentially slow down the runtime by assigning GPU in interleaved ways.
  if (num_iter_ % 2 == 0) {
    std::cout << "Swapped\n" << std::flush;
    target_gpu_id = (target_gpu_id == 1)? 0 : 1;
  }
  */
  Processor target_proc = GPU_procs_[target_gpu_id];
  std::cout << "gpu kind:" << target_proc.kind() << ", "<<
    Processor::TOC_PROC << "\n" << std::flush;
  output.target_procs.push_back(target_proc);
  output.chosen_variant = default_find_preferred_variant(task, ctx,
      true, true, target_proc.kind()).variant;
  output.task_priority = 0;
  output.postmap_task = false;

  //----------------------------------------------------------------------------
  // Check cached mappings, and if any mapping for a task exists, reuse that.
  //----------------------------------------------------------------------------

  const uint64_t task_hash = DefaultMapper::compute_task_hash(task);
  std::pair<TaskID, Processor> cache_key(task.task_id, target_proc);
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping> >::const_iterator
    finder = cached_task_mappings_.find(cache_key);
  bool needs_field_constraint_check{false};
  if (finder != cached_task_mappings_.end()) {
    bool found{false};
    for (std::list<CachedTaskMapping>::const_iterator it =
         finder->second.begin(); it != finder->second.end(); ++it) {
      if ((it->variant == output.chosen_variant) &&
          (it->task_hash == task_hash)) {
        output.chosen_instances = it->mapping;
        found = true;
        break;
      }
    }
    if (found) {
      // If it can immediatley reuse the cached instance, use it.
      if (runtime->acquire_and_filter_instances(ctx, output.chosen_instances)) {
        return;
      }
      // It tries to reuse the cached fields. But the fields could be
      // already invalidated by other task mappings. Before the region is
      // reused, field constraints should be reevaluated.
      needs_field_constraint_check = true;
      DefaultMapper::default_remove_cached_task(ctx, output.chosen_variant,
                                                task_hash, cache_key,
                                                output.chosen_instances);
    }
  }

  //----------------------------------------------------------------------------
  // Find or create physical region instances 
  //----------------------------------------------------------------------------

  std::vector<std::set<FieldID>> target_fields(task.regions.size());
  runtime->filter_instances(ctx, task, output.chosen_variant,
                            output.chosen_instances, target_fields);
  // Get task constraints (These are user-specified constraints at application.
  // These are different from layout constraitns.).
  const TaskLayoutConstraintSet &layout_constraints =
    runtime->find_task_layout_constraints(ctx, task.task_id,
                                          output.chosen_variant);

  // Allocate regions onto the target processors.
  for (uint32_t ri = 0; ri < task.regions.size(); ++ri) {
    const RegionRequirement& rreq = task.regions[ri];
    // Skip empty regions. It requires nothing.
    if ((rreq.privilege == LEGION_NO_ACCESS) ||
        (rreq.privilege_fields.empty()) || target_fields[ri].empty()) {
      continue;
    }

    // Get the first valid constraint for the region[ri].
    // A returned constraint could be invalid, and in this case, return
    // the existing any cached memory, or create new memory as the normal
    // execution path does.
    MemoryConstraint memory_constraint =
      DefaultMapper::find_memory_constraint(ctx, task, output.chosen_variant,
                                            ri);
    // Get the best target memory; In this context, the 'best' means the memory
    // having the highest bandwidth or the RDMA memory having the highest
    // bandwidth if RDMA is the preferrable variant.
    Memory target_memory =
      DefaultMapper::default_policy_select_target_memory(ctx, target_proc,
                                                         task.regions[ri],
                                                         memory_constraint);
    // Get valid instances that the runtime already knows.
    std::vector<PhysicalInstance> valid_instances;
    for (std::vector<PhysicalInstance>::const_iterator
         it = input.valid_instances[ri].begin(),
         ie = input.valid_instances[ri].end(); it != ie; ++it) {
      if (it->get_location() == target_memory) {
        valid_instances.push_back(*it);
      }
    }
    std::set<FieldID> valid_target_fields;
    runtime->filter_instances(ctx, task, ri, output.chosen_variant,
                              valid_instances, valid_target_fields);
    // Detach fields from the target region physical instances.
    // Only detach the regions that were not acquire by other tasks.
    // TODO(hc): slighlty confused yet.
    runtime->acquire_and_filter_instances(ctx, valid_instances);
    output.chosen_instances[ri] = valid_instances;
    target_fields[ri] = valid_target_fields;

    if (valid_target_fields.empty()) { continue; }

    // Create the new instance with necessary fields and their previleges.
    size_t footprint;
    if (!DefaultMapper::default_create_custom_instances(ctx, target_proc,
            target_memory, task.regions[ri], ri, target_fields[ri],
            layout_constraints, needs_field_constraint_check,
            output.chosen_instances[ri], &footprint)) {
      DefaultMapper::default_report_failed_instance_creation(task, ri,
          target_proc, target_memory, footprint);
    }
  }

  //----------------------------------------------------------------------------
  // Cache the mapping result. 
  //----------------------------------------------------------------------------

  // Caching the mapping results.
  std::list<CachedTaskMapping> &map_list = cached_task_mappings_[cache_key];
  map_list.push_back(CachedTaskMapping());
  CachedTaskMapping &cached_result = map_list.back();
  cached_result.task_hash = task_hash;
  cached_result.variant = output.chosen_variant;
  cached_result.mapping = output.chosen_instances;

#if 0
    if (input.valid_instances[ri].empty()) {
      output.chosen_instances[ri].resize(1);
      const LayoutConstraintSet empty_constraints;
      const std::vector<LogicalRegion> empty_regions(1, rreq.region);
      /*
      bool created{false};
      bool ok = runtime->find_or_create_physical_instance(ctx,
          default_policy_select_target_memory(ctx, task.target_proc, rreq),
          empty_constraints, empty_regions, output.chosen_instances[ri].back(),
          created, true);
      if (!ok) {
        printf("Failed to find/create empty instance");
        assert(false);
      }
      */

      // 1) find_memory_constraint: get memory constraint (currently, just memory kind).
      // 2) default_policy_select_target_memory: selects the best memory that has the
      //    maximum bandwidth and maximum free memory size.
      // 3) default_create_custom_instances: iterates index and field spaces, checks
      //    constraints, and creates a new physical region instance.
      size_t footprint;
      std::set<FieldID> copy = task.regions[ri].privilege_fields;
      DefaultMapper::default_create_custom_instances(
          ctx, target_proc,
          DefaultMapper::default_policy_select_target_memory(ctx,
            target_proc,
            task.regions[ri],
            find_memory_constraint(ctx, task, output.chosen_variant, ri)),
          task.regions[ri], ri, copy,
          runtime->find_task_layout_constraints(ctx,
            task.task_id,
            output.chosen_variant), false,
          output.chosen_instances[ri], &footprint);
      continue;
    }
    output.chosen_instances[ri] = input.valid_instances[ri];
    runtime->acquire_and_filter_instances(ctx, output.chosen_instances);
  }
#endif
}

static void create_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs) {
  std::vector<Processor>* procs_list = new std::vector<Processor>();

  std::cout << "[HeatTransferMapper] Create mappers\n" << std::flush;

  // Construct a processor list for a mapper.
  Machine::ProcessorQuery procs_query(machine);
  procs_query.only_kind(Processor::TOC_PROC);
  for (Machine::ProcessorQuery::iterator it = procs_query.begin();
       it != procs_query.end(); ++it) {
    // Each iterator points to a Processor object.
    procs_list->push_back(*it);
    std::cout << "Create a mapper on " <<
                      it->get_kind_name(it->kind()) << "(idx:" <<
                      std::distance(procs_query.begin(), it) << ")\n" << std::flush;
  }
  // Each local processor has a mapper and at the same time, the mapper
  // maintains global processor information.
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); ++it) {
    BridgeMapper* mapper = new BridgeMapper(runtime->get_mapper_runtime(),
                                            machine, *it, "heat_transfer_mapper",
                                            procs_list);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void register_mappers() {
  std::cout << "Register mappers\n" << std::flush;
  Runtime::add_registration_callback(create_mappers);
}

#include "heat_transfer/mappers/bridge_mapper.h"
#include "heat_transfer/profiling/profiling_constants.h"
#include "mappers/default_mapper.h"

#include "legion.h"
#include "legion/legion_mapping.h"

using namespace Legion;
using namespace Legion::Mapping;

// This class is a bridge mapper between application and a RL module.
class BridgeMapper : public DefaultMapper {
public:
  BridgeMapper(MapperRuntime* rt, Machine machine, Processor local_proc,
               const char* mapper_name, std::vector<Processor>* procs_list);

  // ---------------------------------------------------------------------------
  // Function overrides.
  // ---------------------------------------------------------------------------
  virtual void select_task_options(const MapperContext ctx,
                                   const Task& task, TaskOptions& output);

  virtual void map_task(const MapperContext ctx, const Task& task,
                        const MapTaskInput& input, MapTaskOutput& output);

  virtual void report_profiling(const MapperContext ctx, const Task& task,
                                const TaskProfilingInfo& input);

  virtual void select_tunable_value(const MapperContext ctx, const Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);

  virtual void handle_message(const MapperContext ctx,
                              const MapperMessage &msg);

  void generate_mapping_using_coord(const MapperContext ctx,
                                    const Task& task,
                                    const MapTaskInput& input,
                                    MapTaskOutput& output);


  Processor get_fixed_target_proc(const Task& task);
  Memory get_fixed_target_mem(const MapperContext ctx, Processor target_proc,
                              const Task& task, MemoryConstraint mc,
                              uint32_t ri);

  void set_mapping_from(const MapperContext ctx, const Task& task,
                        const MapTaskInput& input, MapTaskOutput& output);

  bool is_start_timer_task(const Task& t);
  bool is_end_timer_task(const Task& t);
  bool is_timer_task(const Task& t);
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
  // CPU processor for auxiliary tasks
  Processor default_cpu_processor_;
  // Task ID to mapping.
  std::map<TaskID, Mapping> sampled_mapping_;
  // Cached mapping.
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping> > cached_task_mappings_;
  // GPU processors.
  // Logical index will be used. TODO(hc): is it corresponding to CUDA number?
  std::vector<Processor> GPU_procs_;
  // Task set for timers.
  std::set<std::string> tasks_start_timer_;
  std::set<std::string> tasks_end_timer_;
  // Timers.
  uint64_t start_time_;
  std::vector<double> end_times_;

  MapperEvent defer_mapping_start_;
  MapperEvent defer_mapping_stop_;
};

BridgeMapper::BridgeMapper(MapperRuntime* rt, Machine machine,
                           Processor local_proc, const char* mapper_name,
                           std::vector<Processor>* procs_list)
  : DefaultMapper(rt, machine, local_proc, mapper_name),
  num_iter_(0), default_cpu_processor_(Processor::NO_PROC) {
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);

  tasks_start_timer_.insert("start_timer");
  tasks_end_timer_.insert("stop_timer");
  
  for (std::set<Processor>::const_iterator it = all_procs.begin();
       it != all_procs.end(); ++it) {
    switch(it->kind()) {
      case Processor::LOC_PROC:
        {
          if (default_cpu_processor_ == Processor::NO_PROC) {
            default_cpu_processor_ = *it;
            printf(" CPU processor ID " IDFMT " \n", it->id);
          }
          break;
        }
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


bool BridgeMapper::is_start_timer_task(const Task& t) {
  return tasks_start_timer_.find(t.get_task_name()) != tasks_start_timer_.end();
}

bool BridgeMapper::is_end_timer_task(const Task& t) {
  return tasks_end_timer_.find(t.get_task_name()) != tasks_end_timer_.end();
}

bool BridgeMapper::is_timer_task(const Task& t) {
  return is_start_timer_task(t) || is_end_timer_task(t);
}

void BridgeMapper::select_task_options(const MapperContext ctx,
                                       const Task& task, TaskOptions& output) {
  printf("Select task options\n");
  std::cout << "Task init procs:" << task.target_proc<< ", no proc:"
    << Processor::NO_PROC << "\n" << std::flush;
  // Eanble the runtime to track, and get valid instances and get premapped
  // region information.
  output.valid_instances = true;
  output.stealable = true;
  output.inline_task = false;
  output.map_locally = false;

  // Necessary to map timer tasks on the same CPU for correctness.
  if (is_timer_task(task)) {
    output.stealable = false;
    output.initial_proc = default_cpu_processor_;
  }
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

void BridgeMapper::handle_message(const MapperContext ctx,
                                  const MapperMessage &msg) {
  std::cout << "handle msg()\n" << std::flush;
}

void BridgeMapper::report_profiling(const MapperContext ctx, const Task& task,
                                    const TaskProfilingInfo& input) {
  std::cout << "report profiling\n" << std::flush;
  using namespace ProfilingMeasurements;
  OperationTimeline *tl =
      input.profiling_responses.get_measurement<OperationTimeline>();
  int64_t started{-1}, ended{-1};
  if (tl) {
    started = tl->start_time;
    ended = tl->complete_time;
    delete tl;
  } else {
  std::cout << "No operation timeline for task " << task.get_task_name() << "\n"
      << std::flush;
    assert(false);
  }

  if (is_start_timer_task(task)) {
    std::cout << "start timer report profiling\n" << std::flush;
    start_time_ = ended;
    std::cout << "start timer event is triggered\n" << std::flush;
    runtime->trigger_mapper_event(ctx, defer_mapping_start_);
  } else if (is_end_timer_task(task)) {
    end_times_.emplace_back((double)(started - start_time_) / 1000000.0);
    std::cout << " [Iteration:" << num_iter_ << "] Time " << end_times_.back()
      << "ms \n" << std::flush;
    std::cout << "stop timer event is triggered\n" << std::flush;
    runtime->trigger_mapper_event(ctx, defer_mapping_stop_);
  }

  num_iter_ += 1;
}

void BridgeMapper::map_task(const MapperContext ctx, const Task& task,
                            const MapTaskInput& input, MapTaskOutput& output) {
  std::cout << "Task mapping: " << task.get_task_name() << ", # of regions:" << task.regions.size() << "\n" << std::flush;
  std::cout << ", task domain point:" << task.index_point << ", target proc:" << task.target_proc << "\n" << std::flush;
  //DefaultMapper::map_task(ctx, task, input, output);

  if (!is_timer_task(task) && strstr(task.get_task_name(), "sweep") == NULL) {
    std::cout << "Default mapper is used. (task name:" << task.get_task_name()
      << ") \n" << std::flush;
    // The first iteration uses the default mapper.
    DefaultMapper::map_task(ctx, task, input, output);
  } else if (is_timer_task(task)) {
    std::cout << "Timer tasks are mapped. (task name:" << task.get_task_name()
      << ") \n" << std::flush;
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.profiling_priority = 1;
    DefaultMapper::map_task(ctx, task, input, output);
  } else {
    std::cout << "Sampled mapping is used. (task name:" << task.get_task_name()
      << ") \n" << std::flush;
    //generate_mapping_using_coord(ctx, task, input, output);
    set_mapping_from(ctx, task, input, output);
  }

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
  // Remove elements from chosen instances if any conflict exists between
  // the layout contraint.
  // The target_fields will be filled with privileged fields that
  // do not exist on the instance yet; so need to be created.
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
#endif
}

/// This function is used to block the execution by using a customized
/// event. For example, the actual processing for start/stop timer tasks
/// are on the mapper layer. The only objective of the application level
/// tasks is to fence the execution and invoke the corresponding information
/// TODO(hc): need to write this down.
/// collection task declared at the mapper.
/// register events, and blocks
/// the execution until 
void BridgeMapper::select_tunable_value(const MapperContext ctx,
                                        const Task& task,
                                        const SelectTunableInput& input,
                                        SelectTunableOutput& output) {
  switch (input.tunable_id) {
    case TUNABLE_VALUE_START: {
      size_t* result = (size_t*) malloc(sizeof(size_t));
      output.value = result;
      output.size = sizeof(size_t);
      runtime->disable_reentrant(ctx);
      if (!defer_mapping_start_.exists()) {
        printf("Creating new start event\n");
        defer_mapping_start_ = runtime->create_mapper_event(ctx);
      }
      runtime->enable_reentrant(ctx);
      std::cout << "Wait on start timer\n" << std::flush;
      runtime->wait_on_mapper_event(ctx, defer_mapping_start_);
      std::cout << "Wait on start timer [done] \n" << std::flush;
      defer_mapping_start_ = MapperEvent();
      *result = TUNABLE_VALUE_START;
      break;
    }
    case TUNABLE_VALUE_STOP: {
      size_t* result = (size_t*) malloc(sizeof(size_t));
      output.value = result;
      output.size = sizeof(size_t);
      runtime->disable_reentrant(ctx);
      if (!defer_mapping_stop_.exists()) {
        printf("Creating new stop event\n");
        defer_mapping_stop_ = runtime->create_mapper_event(ctx);
      }
      runtime->enable_reentrant(ctx);
      std::cout << "Wait on stop timer\n" << std::flush;
      runtime->wait_on_mapper_event(ctx, defer_mapping_stop_);
      std::cout << "Wait on stop timer [done] \n" << std::flush;
      defer_mapping_stop_ = MapperEvent();
      *result = TUNABLE_VALUE_STOP;
      break;
    }
    default: {
    }
  }
}

Processor BridgeMapper::get_fixed_target_proc(const Task& task) {
  uint32_t target_gpu_id = calc_gpu_id(task.index_point, task.index_domain); 
  return GPU_procs_[target_gpu_id];
}

Memory BridgeMapper::get_fixed_target_mem(const MapperContext ctx,
                                          Processor target_proc,
                                          const Task& task, MemoryConstraint mc,
                                          uint32_t ri) {
  return DefaultMapper::default_policy_select_target_memory(ctx, target_proc,
                                                            task.regions[ri],
                                                            mc);
}

void BridgeMapper::set_mapping_from(const MapperContext ctx, const Task& task,
                                    const MapTaskInput& input,
                                    MapTaskOutput& output) {
  //----------------------------------------------------------------------------
  // Choose default option.
  //----------------------------------------------------------------------------

  Processor target_proc = get_fixed_target_proc(task);
  // Ignore existing target processors and always use the specified processor.
  output.target_procs.clear();
  output.target_procs.push_back(target_proc);
  output.chosen_variant = default_find_preferred_variant(task, ctx,
      true, true, target_proc.kind()).variant;
  output.task_priority = 0;
  output.postmap_task = false;

  // TODO(hc): disable the mapping cache feature to use the specified mapping.
  // (The current caching considers any visible memory instance as a candidate)

  //----------------------------------------------------------------------------
  // Find or create physical region instances 
  //----------------------------------------------------------------------------

  std::vector<std::set<FieldID>> target_fields(task.regions.size());
  // Remove elements from chosen instances if any conflict exists between
  // the layout contraint.
  // The target_fields will be filled with privileged fields that
  // do not exist on the instance yet; so need to be created.
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
    // Get the specified memory instace. Note that the current implementation
    // uses the default policy. This should be the specified memory
    // by the RL module.
    Memory target_memory = get_fixed_target_mem(ctx, target_proc, task,
                                                memory_constraint, ri);

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
            layout_constraints, false,
            output.chosen_instances[ri], &footprint)) {
      DefaultMapper::default_report_failed_instance_creation(task, ri,
          target_proc, target_memory, footprint);
    }
  }
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

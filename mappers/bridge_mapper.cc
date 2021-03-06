#include "2D_heat_transform/mappers/bridge_mapper.h"
#include "2D_heat_transform/profiling/profiling_constants.h"
#include "mappers/default_mapper.h"

#include "legion.h"
#include "legion/legion_mapping.h"

using namespace Legion;
using namespace Legion::Mapping;

/// This class is a bridge mapper between application and a RL module.
class BridgeMapper : public DefaultMapper {
public:
  BridgeMapper(MapperRuntime* rt, Machine machine, Processor local_proc,
               const char* mapper_name, std::vector<Processor>* procs_list);

  /// --------------------------------------------------------------------------
  /// Overridden functions.
  /// --------------------------------------------------------------------------
  virtual void select_task_options(const MapperContext ctx,
                                   const Task& task, TaskOptions& output);

  virtual void map_task(const MapperContext ctx, const Task& task,
                        const MapTaskInput& input, MapTaskOutput& output);

  virtual void report_profiling(const MapperContext ctx, const Task& task,
                                const TaskProfilingInfo& input);

  virtual void select_tunable_value(const MapperContext ctx, const Task& task,
                                    const SelectTunableInput& input,
                                    SelectTunableOutput& output);

  /// --------------------------------------------------------------------------
  /// Other functions.
  /// --------------------------------------------------------------------------
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
  /// Struct to store a generated mapping.
  /// Instances of it are generated by an external module.
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
  /// Number of epochs for training.
  size_t num_epochs_;
  /// Fixed CPU processor for auxiliary tasks.
  Processor default_cpu_processor_;
  /// Task ID to a mapping result.
  std::map<TaskID, Mapping> sampled_mapping_;
  /// Cached mapping.
  std::map<std::pair<TaskID, Processor>,
           std::list<CachedTaskMapping> > cached_task_mappings_;
	/// Track a mapping between a processor and the number of tasks.
	std::map<Processor, size_t> num_readytasks_per_proc_;
  /// Track a mapping between a processor and a pair of region tree and
  /// the number of ready tasks.
  std::map<Processor, std::map<RegionTreeID, size_t>>
           num_regiontrees_per_proc_;

  /// GPU processors.
  /// Logical index will be used.
  /// TODO(hc): is it corresponding to CUDA number?
  std::vector<Processor> GPU_procs_;
  /// Task set for timers.
  std::set<std::string> tasks_start_timer_;
  std::set<std::string> tasks_end_timer_;
  /// Timers.
  uint64_t start_time_;
  std::vector<double> end_times_;
  /// Events for timers.
  MapperEvent defer_mapping_start_;
  MapperEvent defer_mapping_stop_;
  /// Enable a debug mode if it sets true.
  bool debug_;
};

static void parse_args(bool& debug) {
  int argc = HighLevelRuntime::get_input_args().argc;
  char **argv = HighLevelRuntime::get_input_args().argv;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-bm:debug")) {
      debug = true;
      break;
    }
  }
}

BridgeMapper::BridgeMapper(MapperRuntime* rt, Machine machine,
                           Processor local_proc, const char* mapper_name,
                           std::vector<Processor>* procs_list)
  : DefaultMapper(rt, machine, local_proc, mapper_name),
  num_epochs_(0), default_cpu_processor_(Processor::NO_PROC), debug_(true) {
  std::set<Processor> all_procs;
  machine.get_all_processors(all_procs);

  parse_args(debug_);

  // Add and track timer tasks.
  tasks_start_timer_.insert("start_timer");
  tasks_end_timer_.insert("stop_timer");
  
  for (std::set<Processor>::const_iterator it = all_procs.begin();
       it != all_procs.end(); ++it) {
    switch(it->kind()) {
      case Processor::LOC_PROC:
        {
          // Use a single CPU to run auxiliary tasks for profiling.
          if (default_cpu_processor_ == Processor::NO_PROC) {
            default_cpu_processor_ = *it;
            if (debug_) {
              printf(" CPU processor ID " IDFMT " \n", it->id);
            }
          }
          break;
        }
      case Processor::TOC_PROC:
        {
          GPU_procs_.emplace_back(*it);
          if (debug_) {
            printf(" GPU processor ID " IDFMT " \n", it->id);
          }
          break;
        }
      default:
        break;
    }
  }  
}

/// Check if a task t is a start timer task or not.
bool BridgeMapper::is_start_timer_task(const Task& t) {
  return tasks_start_timer_.find(t.get_task_name()) != tasks_start_timer_.end();
}

/// Check if a task t is a stop timer task or not.
bool BridgeMapper::is_end_timer_task(const Task& t) {
  return tasks_end_timer_.find(t.get_task_name()) != tasks_end_timer_.end();
}

/// Check if a task t is a timer task.
bool BridgeMapper::is_timer_task(const Task& t) {
  return is_start_timer_task(t) || is_end_timer_task(t);
}

void BridgeMapper::select_task_options(const MapperContext ctx,
                                       const Task& task, TaskOptions& output) {
  if (debug_) {
    std::cout << "Task init procs:" << task.target_proc<< ", no proc:"
      << Processor::NO_PROC << "\n" << std::flush;
  }
  // Eanble the runtime to track, and get valid instances and get premapped
  // region information.
  output.valid_instances = true;
  // TODO(hc): still don't know what this is.
  output.inline_task = false;

  if (is_timer_task(task)) {
    // Profiling tasks should not be stealed.
    output.stealable = false;
    // Profiling tasks should run on the same CPU processor for correctness.
    output.initial_proc = default_cpu_processor_;
  } else {
    output.stealable = true;
    output.map_locally = false;
  }
}

void BridgeMapper::report_profiling(const MapperContext ctx, const Task& task,
                                    const TaskProfilingInfo& input) {
  if (debug_) {
    std::cout << "[Report profiling] Report profiling starts.\n" << std::flush;
  }
  using namespace ProfilingMeasurements;

	if (is_timer_task(task)) {
		OperationTimeline *tl =
				input.profiling_responses.get_measurement<OperationTimeline>();
		int64_t started{-1}, ended{-1};
		if (tl) {
			started = tl->start_time;
			ended = tl->complete_time;
			delete tl;
		} else {
			std::cout << "No operation timeline for task " << task.get_task_name() <<
				"\n" << std::flush;
			assert(false);
		}

		if (is_start_timer_task(task)) {
			if (debug_) {
				std::cout << "[Report profiling] Start timer report profiling.\n"
					<< std::flush;
			}
			start_time_ = ended;
			if (debug_) {
				std::cout << "[Rerport profiling] start timer event is triggered.\n"
					<< std::flush;
			}
			// Trigger the timer start event and wake the task up that has been
			// waiting for the event.
			runtime->trigger_mapper_event(ctx, defer_mapping_start_);
		} else if (is_end_timer_task(task)) {
			end_times_.emplace_back((double)(started - start_time_) / 1000000.0);
			if (debug_) {
				std::cout << "[Report profiling] [Iteration:" << num_epochs_ << "] Time "
					<< end_times_.back() << "ms \n" << std::flush;
				std::cout << "[Report profiling] stop timer event is triggered\n" <<
					std::flush;
			}
			runtime->trigger_mapper_event(ctx, defer_mapping_stop_);
		}
	} else {
    // Track the number of ready tasks.
    const Processor& target_proc = task.current_proc;
    if (num_readytasks_per_proc_.find(target_proc) != num_readytasks_per_proc_.end()) {
      num_readytasks_per_proc_[target_proc] -= 1;
    } else {
      std::cout << "Target processor does not run the specified task.\n" << std::flush;
      assert(false);
    }
    std::cout << "[-] Num of tasks for a processor [" << target_proc << "]:"
      << num_readytasks_per_proc_[target_proc] << "\n" << std::flush;

    // Update the number of region-task mapping.
    for (uint32_t ri = 0; ri < task.regions.size(); ++ri) {
      LogicalRegion logical_region = task.regions[ri].region;
      RegionTreeID tree_id = logical_region.get_tree_id();
      num_regiontrees_per_proc_[target_proc][tree_id] -= 1;
      std::cout << "Decreased " << tree_id << "\n";
      if (num_regiontrees_per_proc_[target_proc][tree_id] < 0) {
        std::cout << "A region of a task was not tracked.\n" << std::flush;
        assert(false);
      }
    }
  }

  num_epochs_ += 1;
}

void BridgeMapper::map_task(const MapperContext ctx, const Task& task,
                            const MapTaskInput& input, MapTaskOutput& output) {
  if (debug_) {
    std::cout << "[Mapping task] Task mapping: " << task.get_task_name() <<
      ", # of regions:" << task.regions.size() << "\n" << std::flush;
    std::cout << ", task domain point:" << task.index_point <<
      ", target proc:" << task.target_proc << "\n" << std::flush;
  }

  if (!is_timer_task(task) && strstr(task.get_task_name(), "sweep") == NULL) {
    // Other than sweep task, uses the default mapper.
    if (debug_) {
      std::cout << "[Mapping task] Default mapper is used. (task name:" <<
        task.get_task_name() << ") \n" << std::flush;
    }
    DefaultMapper::map_task(ctx, task, input, output);
  } else if (is_timer_task(task)) {
    // For timer tasks, register profiling types to measure, and
    // uses the default mapper.
    if (debug_) {
      std::cout << "[Mapping task] Timer tasks are mapped. (task name:" <<
        task.get_task_name() << ") \n" << std::flush;
    }
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.profiling_priority = 1;
    DefaultMapper::map_task(ctx, task, input, output);
  } else {
    // Uses a mapping generated from the external library.
    if (debug_) {
      std::cout << "[Mapping task] Sampled mapping is used. (task name:" <<
        task.get_task_name() << ") \n" << std::flush;
    }
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    //generate_mapping_using_coord(ctx, task, input, output);
    set_mapping_from(ctx, task, input, output);
  }

	// Track the number of ready tasks.
  Processor& target_proc = output.target_procs.back();
	if (num_readytasks_per_proc_.find(target_proc) != num_readytasks_per_proc_.end()) {
		num_readytasks_per_proc_[target_proc] += 1;
	} else {
		num_readytasks_per_proc_[target_proc] = 1;
	}
	std::cout << "Num of tasks for a processor [" << target_proc << "]:"
		<< num_readytasks_per_proc_[target_proc] << "\n" << std::flush;
}

/// It calculates a gpu id by using a domain point.
/// It is a test code and has a very strong assumption that the
/// current setup only uses 2 GPUs. If a coordination is close to the left
/// part of the domain space, assigns GPU0. Otherwise, assigns GPU1.
static uint32_t calc_gpu_id(DomainPoint dp, Domain dm) {
  // Simplistic policy: ([0:N], [0:B-1]) ([0:N], [B:N]) parititoning.
  uint32_t block_size = dm.hi().point_data[0] / 2;
  uint32_t gpu_id = 0;
  if (dp.point_data[1] > block_size) {
    gpu_id = 1;
  }
  return gpu_id;
}

/// This function maps a task based on its domain point.
/// This is not practical mapping, but was written as a practice to prepare
/// a mapping generated from outer modules. For example, our ultimate goal is to
/// use the target processor and target memory chosen by external modules (like
/// RL trainer). The point of this mapper is to adopt equivalent optimizations
/// like caching to the default mapper, except REDUCE and inner task
/// specializations.
/// (TODO(hc): I still don't know what the inner task is)
void BridgeMapper::generate_mapping_using_coord(const MapperContext ctx,
                                                const Task& task,
                                                const MapTaskInput& input,
                                                MapTaskOutput& output) {
  //----------------------------------------------------------------------------
  // Choose default option.
  //----------------------------------------------------------------------------

  uint32_t target_gpu_id = calc_gpu_id(task.index_point, task.index_domain); 
  Processor target_proc = GPU_procs_[target_gpu_id];
  if (debug_) {
    std::cout << task.index_point[0] << "," << task.index_point[1] << 
      " on " << task.index_domain.lo() << "-" << task.index_domain.hi() <<
      " is assigned to GPU" << target_gpu_id <<
      "\n" << std::flush;
  }
  output.target_procs.push_back(target_proc);
  output.chosen_variant = default_find_preferred_variant(task, ctx,
      true, true, target_proc.kind()).variant;
  // No meaning at this point.
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
  // Remove an instance from chosen instances if any layout constraint conflict
  // exists. (IIUC, there are two types of constraints: task layout constraint
  // and existing layout constraint.)
  //
  // After checks the layout constraints, filter_instances() checks field
  // space for each layout.
  // If field constraints do not match, like layout constraint comparisons,
  // remove the existing field instances, and adds the fields to the
  // `target_fields`. (So to speak, the `target_fields` implies that we
  // need to create the fields on that with proper privileges again.)
  // If that is the case, create the new fields with the proper privileges.
  // Regarding the proper 'privilege', in this context, it could mean
  // that either 1) privileges of the task constraint and the existing layout
  // exactly match on the exact-match mode, or 2) literally inclusive relations.
  runtime->filter_instances(ctx, task, output.chosen_variant,
                            output.chosen_instances, target_fields);
  // Get task constraints (These are user-specified constraints at application.
  // These are different from the existing layout constraitns.).
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
    // Get valid instances that the runtime already knows (premapped regions).
    std::vector<PhysicalInstance> valid_instances;
    for (std::vector<PhysicalInstance>::const_iterator
         it = input.valid_instances[ri].begin(),
         ie = input.valid_instances[ri].end(); it != ie; ++it) {
      if (it->get_location() == target_memory) {
        valid_instances.push_back(*it);
      }
    }
    // TODO(hc): 
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

/// This function just returns one of the GPUs.
/// It is not a practical processor choosing function, but acts
/// as the external library.
Processor BridgeMapper::get_fixed_target_proc(const Task& task) {
  uint32_t target_gpu_id = calc_gpu_id(task.index_point, task.index_domain); 
  return GPU_procs_[target_gpu_id];
}

/// This function just returns one of the visible memory from.
/// the assigned processor.
/// It is not a practical processor choosing function, but acts
/// as the external library.
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
  std::cout << "Set mapping from..\n" << std::flush;
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

  // --- Test code getting the number of existing physical instances -----------

  // Check if a processor is materialized on the proc. to region tree mapping.
  if (num_regiontrees_per_proc_.find(target_proc) ==
      num_regiontrees_per_proc_.end()) {
    num_regiontrees_per_proc_[target_proc] = {};
  }

  size_t accum_tasks_using_common_regions{0};
  // First, get the number of existing physical instances as a test.
  for (uint32_t ri = 0; ri < task.regions.size(); ++ri) {
    const TaskLayoutConstraintSet &task_layout_constraints =
      runtime->find_task_layout_constraints(ctx,
                            task.task_id, output.chosen_variant);
    // Iterate layout constraints specified to a task.
    // Different layouts having different constraints create different
    // instances.
    for (std::multimap<unsigned, LayoutConstraintID>::const_iterator
           lay_const_it = task_layout_constraints.layouts.lower_bound(ri);
           lay_const_it != task_layout_constraints.layouts.upper_bound(ri);
           ++lay_const_it) {
      // Find a set of constraints for a index space.
      const LayoutConstraintSet& index_constraints =
        runtime->find_layout_constraints(ctx, lay_const_it->second);

      // Iterate all visible memory from the chosen processor (by action
      // chosen by RL) and check if it materializes a region.
      uint64_t num_existing_regions{0};
      Machine::MemoryQuery visible_memories(machine);
      for (Machine::MemoryQuery::iterator visible_mem_it =
            visible_memories.begin();
            visible_mem_it != visible_memories.end(); ++visible_mem_it) {
        Memory target_memory = *visible_mem_it;
        // Return the root logical region of the target region.
        LogicalRegion target_region =
          DefaultMapper::default_policy_select_instance_region(ctx, target_memory,
              task.regions[ri], index_constraints, false /* force new instance */,
              true);
        std::vector<LogicalRegion> target_regions(1, target_region);

        // First, compare memory constraints and region constraints.
        const MemoryConstraint& mem_constriant =
          runtime->find_layout_constraints(ctx, lay_const_it->second).
                   memory_constraint;
        if (mem_constriant.is_valid() &&
            mem_constriant.get_kind() != visible_mem_it->kind()) { continue; }
        PhysicalInstance target_instance;
        bool tight_region_bounds =
          index_constraints.specialized_constraint.is_exact() ||
          ((task.regions[ri].tag & DefaultMapper::EXACT_REGION) != 0);
        // Second, check if a physical instance exists.
        if (runtime->find_physical_instance(ctx, target_memory,
              index_constraints, target_regions, target_instance, true,
              tight_region_bounds)) {
          ++num_existing_regions;
        }
      }
      std::cout << "Region index: " << ri << " is instantiated on " <<
        num_existing_regions << " memory\n";
    }

    // Get the logical region.
    LogicalRegion logical_region = task.regions[ri].region;
    // Get the logical region tree id.
    RegionTreeID tree_id = logical_region.get_tree_id(); 


    std::map<RegionTreeID, size_t>::iterator found;
    if ((found = num_regiontrees_per_proc_[target_proc].find(tree_id)) !=
          num_regiontrees_per_proc_[target_proc].end()) {
      std::cout << "Found.:" << found->second << "," << tree_id << "\n";
      accum_tasks_using_common_regions += found->second;
      found->second += 1;
    } else {
      std::cout << "Not found:" << tree_id << "\n";
      num_regiontrees_per_proc_[target_proc][tree_id] = 1;
    }
  }

  std::cout << "Task:" << task.get_task_name() << "'s regions are materialized "
    << " by " << accum_tasks_using_common_regions << " ready tasks.\n";

#if 0
  using Legion::Internal::ProcessorManager;
  // Second, access processor manager, and get the number of tasks for each processor.
  std::map<Processor, ProcessorManager*>& proc_managers =
                                                runtime->get_proc_managers(ctx);
  for (std::map<Processor, ProcessorManager*>::const_iterator it =
        proc_managers.begin(); it != proc_managers.end(); ++it) {
    it->second->check_task_on_ready_queue(0);
  }
#endif

  // ---------------------------------------------------------------------------
    
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
    // It usually can be known due to premapping.
    std::vector<PhysicalInstance> valid_instances;
    for (std::vector<PhysicalInstance>::const_iterator
         it = input.valid_instances[ri].begin(),
         ie = input.valid_instances[ri].end(); it != ie; ++it) {
      if (it->get_location() == target_memory) {
        valid_instances.push_back(*it);
      }
    }
    std::set<FieldID> valid_target_fields;
    // TODO(hc): need to understand the difference between the filter_instance()
    // at the outside of this loop.
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

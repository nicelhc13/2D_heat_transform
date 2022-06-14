#include "2d_heat_transfer_mapper.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

static LegionRuntime::Logger::Category log_heattransform("2d_heat_trans");

class HeatTransferMapper : public DefaultMapper
{
public:
  HeatTransferMapper(MapperRuntime *rt, Machine machine, Processor local,
                     const char *mapper_name,
                     std::vector<Processor>* procs_list);

  virtual void select_task_options(const MapperContext ctx,
                                   const Task&         task,
                                         TaskOptions&  output);

  virtual void map_task(const MapperContext  ctx,
                        const Task&          task,
                        const MapTaskInput&  input,
                              MapTaskOutput& output);

  virtual Processor default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task);

  virtual void default_policy_select_target_processors(
                                    MapperContext ctx,
                                    const Task &task,
                                    std::vector<Processor> &target_procs);

  /*
  virtual const std::map<VariantID, Processor::Kind>& find_task_variants(
                                                MapperContext ctx, TaskID task_id);
                                                */

	virtual void map_constrained_requirement(MapperContext ctx,
			const RegionRequirement &req, MappingKind mapping_kind,
			const std::vector<LayoutConstraintID> &constraints,
			std::vector<PhysicalInstance> &chosen_instances, Processor restricted);



	virtual void map_random_requirement(MapperContext ctx,
    																  const RegionRequirement &req,
    												 					std::vector<PhysicalInstance> &chosen_instances,
														 					Processor restricted);

private:
  std::vector<Processor>& procs_list;
};


HeatTransferMapper::HeatTransferMapper(MapperRuntime *rt, Machine machine,
                                       Processor local, const char *mapper_name, 
                                       std::vector<Processor>* _procs_list)
  : DefaultMapper(rt, machine, local, mapper_name), procs_list(*_procs_list) {
  std::cout << "Constructor \n";
}

Processor HeatTransferMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  std::cout << "Default policy select initial processor \n";
  return DefaultMapper::default_policy_select_initial_processor(ctx, task);
}

void HeatTransferMapper::select_task_options(const MapperContext ctx,
                                             const Task& task,
                                             TaskOptions& output) {
  std::cout << "Select task options \n";
  output.initial_proc = default_policy_select_initial_processor(ctx, task);
  output.inline_task = false;
  output.stealable = false;
  output.map_locally = false;
}

/*
const std::map<VariantID, Processor::Kind>& HeatTransferMapper::find_task_variants(
                                                MapperContext ctx, TaskID task_id) {
  std::map<TaskID, std::map<VariantID, Processor::Kind>> variant_processor_kinds;
  std::map<TaskID, std::map<VariantID, Processor::Kind>>::const_iterator
    finder = variant_processor_kinds.find(task_id);
  if (finder != variant_processor_kinds.end()) {
    return finder->second;
  }
  std::vector<VariantID> valid_variants;
  runtime->find_valid_variants(ctx, task_id, valid_variants);

	std::cout << "Task ID:" << task_id << "\n";
	std::cout << "Valid variant size:" << valid_variants.size() << "\n";
  std::map<VariantID, Processor::Kind> kinds;
  for (std::vector<VariantID>::const_iterator it = valid_variants.begin();
        it != valid_variants.end(); ++it) {
    const ExecutionConstraintSet &constraints =
      runtime->find_execution_constraints(ctx, task_id, *it);
    if (constraints.processor_constraint.is_valid())
      kinds[*it] = constraints.processor_constraint.valid_kinds[0];
    else
      kinds[*it] = Processor::LOC_PROC;
  }
  std::map<VariantID, Processor::Kind> &result =
    variant_processor_kinds[task_id];
  result = kinds;
	std::cout << "Valid variant size:" << valid_variants.size() << " [done] \n";
  return result;
}
*/

void HeatTransferMapper::map_constrained_requirement(MapperContext ctx,
    const RegionRequirement &req, MappingKind mapping_kind,
    const std::vector<LayoutConstraintID> &constraints,
    std::vector<PhysicalInstance> &chosen_instances, Processor restricted) {
  chosen_instances.resize(constraints.size());
  unsigned output_idx = 0;
  for (std::vector<LayoutConstraintID>::const_iterator lay_it =
        constraints.begin(); lay_it != constraints.end(); lay_it++, output_idx++) {
    const LayoutConstraintSet &layout_constraints =
      runtime->find_layout_constraints(ctx, *lay_it);
    Machine::MemoryQuery all_memories(machine);
    if (restricted.exists())
      all_memories.has_affinity_to(restricted);
    // This could be a big data structure in a big machine
    std::map<unsigned,Memory> random_memories;
    for (Machine::MemoryQuery::iterator it = all_memories.begin();
          it != all_memories.end(); it++)
    {
      random_memories[default_generate_random_integer()] = *it;
    }
    bool made_instance = false;
    while (!random_memories.empty())
    {
      std::map<unsigned,Memory>::iterator it = random_memories.begin();
      Memory target = it->second;
      random_memories.erase(it);
      if (target.capacity() == 0)
        continue;
      if (default_make_instance(ctx, target, layout_constraints,
            chosen_instances[output_idx], mapping_kind,
            true/*force new*/, false/*meets*/, req))
      {
        made_instance = true;
        break;
      }
    }
    if (!made_instance)
    {
      assert(false);
    }
  }
}

void HeatTransferMapper::map_random_requirement(MapperContext ctx,
    const RegionRequirement &req,
    std::vector<PhysicalInstance> &chosen_instances, Processor restricted) {

  std::vector<LogicalRegion> regions(1, req.region);
  chosen_instances.resize(req.privilege_fields.size());
  unsigned output_idx = 0;
  // Iterate over all the fields and make a separate instance and
  // put it in random places
  for (std::set<FieldID>::const_iterator it = req.privilege_fields.begin();
        it != req.privilege_fields.end(); it++, output_idx++)
  {
    std::vector<FieldID> field(1, *it);
    // Try a bunch of memories in a random order until we find one 
    // that succeeds
    Machine::MemoryQuery all_memories(machine);
    if (restricted.exists())
      all_memories.has_affinity_to(restricted);
    // This could be a big data structure in a big machine
    std::map<unsigned,Memory> random_memories;
    for (Machine::MemoryQuery::iterator it = all_memories.begin();
          it != all_memories.end(); it++)
    {
      random_memories[default_generate_random_integer()] = *it;
    }
    bool made_instance = false;
    while (!random_memories.empty())
    {
      std::map<unsigned,Memory>::iterator it = random_memories.begin();
      Memory target = it->second;
      random_memories.erase(it);
      if (target.capacity() == 0)
        continue;
      // TODO: put in arbitrary constraints to mess with the DMA system
      LayoutConstraintSet constraints;
      default_policy_select_constraints(ctx, constraints, target, req);
      // Overwrite the field constraints 
      constraints.field_constraint = FieldConstraint(field, false);
      // Try to make the instance, we always make new instances to
      // generate as much data movement and dependence analysis as
      // we possibly can, it will also stress the garbage collector
      if (runtime->create_physical_instance(ctx, target, constraints,
                               regions, chosen_instances[output_idx]))
      {
        made_instance = true;
        break;
      }
    }
    if (!made_instance)
    {
      assert(false);
    }
  }
}


void HeatTransferMapper::map_task(const MapperContext ctx, const Task& task,
                           const MapTaskInput& input, MapTaskOutput& output) {
  /*
  const std::map<VariantID,Processor::Kind> &variant_kinds =
    find_task_variants(ctx, task.task_id);
  std::vector<VariantID> variants;
  for (std::map<VariantID,Processor::Kind>::const_iterator it =
        variant_kinds.begin(); it != variant_kinds.end(); it++) {
    if (task.target_proc.kind() == it->second)
      variants.push_back(it->first);
  }
  assert(!variants.empty());
  if (variants.size() > 1) {
    int chosen = default_generate_random_integer() % variants.size();
    output.chosen_variant = variants[chosen];
  }
  else
    output.chosen_variant = variants[0];
  output.target_procs.push_back(task.target_proc);
  std::vector<bool> premapped(task.regions.size(), false);
  for (unsigned idx = 0; idx < input.premapped_regions.size(); idx++) {
    unsigned index = input.premapped_regions[idx];
    output.chosen_instances[index] = input.valid_instances[index];
    premapped[index] = true;
  }
  const TaskLayoutConstraintSet &layout_constraints =
    runtime->find_task_layout_constraints(ctx, task.task_id,
                                          output.chosen_variant);
  for (unsigned idx = 0; idx < task.regions.size(); idx++) {
    if (premapped[idx])
      continue;
    if (task.regions[idx].is_restricted()) {
      output.chosen_instances[idx] = input.valid_instances[idx];
      continue;
    }
    if (layout_constraints.layouts.find(idx) !=
          layout_constraints.layouts.end()) {
      std::vector<LayoutConstraintID> constraints;
      for (std::multimap<unsigned,LayoutConstraintID>::const_iterator it =
            layout_constraints.layouts.lower_bound(idx); it !=
            layout_constraints.layouts.upper_bound(idx); it++)
        constraints.push_back(it->second);
      map_constrained_requirement(ctx, task.regions[idx], TASK_MAPPING,
          constraints, output.chosen_instances[idx], task.target_proc);
    }
    else
      map_random_requirement(ctx, task.regions[idx],
                             output.chosen_instances[idx],
                             task.target_proc);
  }
  output.task_priority = default_generate_random_integer();

  {
    using namespace ProfilingMeasurements;
    output.task_prof_requests.add_measurement<OperationStatus>();
    output.task_prof_requests.add_measurement<OperationTimeline>();
    output.task_prof_requests.add_measurement<RuntimeOverhead>();
  }
  */
  DefaultMapper::map_task(ctx, task, input, output);
}

static void create_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs) {
  std::vector<Processor>* procs_list = new std::vector<Processor>();

  std::cout <<"Create mappers\n";

  // Construct a processor list for a mapper.
  Machine::ProcessorQuery procs_query(machine);
  procs_query.only_kind(Processor::LOC_PROC);
  for (Machine::ProcessorQuery::iterator it = procs_query.begin();
       it != procs_query.end(); ++it) {
    // Each iterator points to a Processor object.
    procs_list->push_back(*it);
    std::cout << std::distance(procs_query.begin(), it) <<
      " -> " << it->get_kind_name(it->kind()) << "\n";
  }

  // Each local processor has a mapper and at the same time, the mapper
  // maintains global processor information.
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); ++it) {
    HeatTransferMapper* mapper = new HeatTransferMapper(runtime->get_mapper_runtime(),
                                            machine, *it, "heat_transfer_mapper",
                                            procs_list);
    runtime->replace_default_mapper(mapper, *it);
  }
}

void HeatTransferMapper::default_policy_select_target_processors(                        
                                    MapperContext ctx,                              
                                    const Task &task,                               
                                    std::vector<Processor> &target_procs)           
{                                                                                   
  target_procs.push_back(task.target_proc);                                         
}                                                                                   

void register_mappers() {
  std::cout << "Register mappers\n";
  Runtime::add_registration_callback(create_mappers);
}

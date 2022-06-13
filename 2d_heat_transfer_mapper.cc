#include "2d_heat_transfer_mapper.h"

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

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

private:
  std::vector<Processor>& procs_list;
};


HeatTransferMapper::HeatTransferMapper(MapperRuntime *rt, Machine machine,
                                       Processor local, const char *mapper_name, 
                                       std::vector<Processor>* _procs_list)
  : DefaultMapper(rt, machine, local, mapper_name), procs_list(*_procs_list) {
  std::cout << "Constructor \n";
}


void HeatTransferMapper::select_task_options(const MapperContext ctx,
                                             const Task& task,
                                             TaskOptions& output) {
  std::cout << "Select task options \n";
}

void HeatTransferMapper::map_task(const MapperContext ctx, const Task& task,
                           const MapTaskInput& input, MapTaskOutput& output) {
  std::cout << "Map tasks \n";
}

static void create_mappers(Machine machine, Runtime *runtime,
                           const std::set<Processor> &local_procs) {
  std::vector<Processor>* procs_list = new std::vector<Processor>();

  // Construct a processor list for a mapper.
  Machine::ProcessorQuery procs_query(machine);
  procs_query.only_kind(Processor::LOC_PROC);
  for (Machine::ProcessorQuery::iterator it = procs_query.begin();
       it != procs_query.end(); ++it) {
    // Each iterator points to a Processor object.
    procs_list->push_back(*it);
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

void register_mappers() {
  Runtime::add_registration_callback(create_mappers);
}

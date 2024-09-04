#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <array>
#include <optional>

#include <arbor/context.hpp>
#include <arbor/common_types.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/profile/meter_manager.hpp>
#include <arbor/profile/profiler.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>
#include <arbor/version.hpp>
#include <arborenv/default_env.hpp>
#include <arborenv/gpu_env.hpp>
#include <arborio/label_parse.hpp>
#include <arborenv/with_mpi.hpp>
#include <arbor/load_balance.hpp>

#include <mpi.h>
#include <nlohmann/json.hpp>

template <typename T>
std::optional<T> find_and_remove_json(const char* name, nlohmann::json& j) {
    auto it = j.find(name);
    if (it==j.end()) {
        return std::nullopt;
    }
    T value = std::move(*it);
    j.erase(name);
    return std::move(value);
}

template <typename T>
void param_from_json(T& x, const char* name, nlohmann::json& j) {
    if (auto o = find_and_remove_json<T>(name, j)) {
        x = *o;
    }
}

template <typename T, size_t N>
void param_from_json(std::array<T, N>& x, const char* name, nlohmann::json& j) {
    std::vector<T> y;
    if (auto o = find_and_remove_json<std::vector<T>>(name, j)) {
        y = *o;
        if (y.size()!=N) {
            throw std::runtime_error("parameter "+std::string(name)+" requires "+std::to_string(N)+" values");
        }
        std::copy(y.begin(), y.end(), x.begin());
    }
}


// Parameters used to generate the random cell morphologies.
struct cell_parameters {
    cell_parameters() = default;

    // Use complex cell or generic cell
    bool complex_cell = false;

    //  Maximum number of levels in the cell (not including the soma)
    unsigned max_depth = 5;

    // The following parameters are described as ranges.
    // The first value is at the soma, and the last value is used on the last level.
    // Values at levels in between are found by linear interpolation.
    std::array<double,2> branch_probs = {1.0, 0.5}; //  Probability of a branch occuring.
    std::array<unsigned,2> compartments = {20, 2};  //  Compartment count on a branch.
    std::array<double,2> lengths = {200, 20};       //  Length of branch in μm.

    // The number of synapses per cell.
    unsigned synapses = 1;
};

struct ring_params {
    ring_params() = default;

    std::string name = "default";
    unsigned num_cells = 5000;
    unsigned ring_size = 5;
    double min_delay = 5;
    double duration = 200;
    double dt = 0.025;
    float event_weight = 0.05;
    std::string odir = ".";
    cell_parameters cell;
};

ring_params read_options(int argc, char** argv) {
    const char* usage = "Usage:  arbor-busyring [params [opath]]\n\n"
                        "Driver for the Arbor busyring benchmark\n\n"
                        "Options:\n"
                        "   params: JSON file with model parameters.\n"
                        "   opath: output path.\n";

    ring_params params;
    if (argc<2) {
        return params;
    }
    if (argc>3) {
        std::cout << usage << std::endl;
        throw std::runtime_error("More than two command line options is not permitted.");
    }

    // Assume that the first argument is a json parameter file
    std::string fname = argv[1];
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    f >> json;

    param_from_json(params.name, "name", json);
    param_from_json(params.num_cells, "num-cells", json);
    param_from_json(params.ring_size, "ring-size", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.dt, "dt", json);
    param_from_json(params.min_delay, "min-delay", json);
    param_from_json(params.event_weight, "event-weight", json);
    param_from_json(params.cell.complex_cell, "complex", json);
    param_from_json(params.cell.max_depth, "depth", json);
    param_from_json(params.cell.branch_probs, "branch-probs", json);
    param_from_json(params.cell.compartments, "compartments", json);
    param_from_json(params.cell.lengths, "lengths", json);
    param_from_json(params.cell.synapses, "synapses", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    // Set optional output path if a second argument was passed
    if (argc==3) {
        params.odir = argv[2];
    }

    return params;
}


using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_kind;
using arb::time_type;
using arb::cable_probe_membrane_voltage;

using namespace arborio::literals;
namespace U = arb::units;

// Generate a cell.
arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params);
arb::cable_cell complex_cell(arb::cell_gid_type gid, const cell_parameters& params);

struct ring_recipe: public arb::recipe {
    ring_recipe(ring_params params):
        num_cells_(params.num_cells),
        min_delay_(params.min_delay),
        event_weight_(params.event_weight),
        params_(params) {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.catalogue.import(arb::global_allen_catalogue(), "");

        if (params.cell.complex_cell) {
            gprop.default_parameters.reversal_potential_method["ca"] = "nernst/ca";
            gprop.default_parameters.axial_resistivity = 100;
            gprop.default_parameters.temperature_K = 34 + 273.15;
            gprop.default_parameters.init_membrane_potential = -90;
        }
    }

    std::any get_global_properties(cell_kind kind) const override { return gprop; }
    cell_size_type num_cells() const override { return num_cells_; }
    cell_kind get_cell_kind(cell_gid_type gid) const override { return cell_kind::cable; }
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        if (params_.cell.complex_cell) {
            return complex_cell(gid, params_.cell);
        }
        return branch_cell(gid, params_.cell);
    }

    // Each cell has one incoming connection, from cell with gid-1,
    // and fan_in-1 random connections with very low weight.
    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<arb::cell_connection> cons;
        const auto ncons = params_.cell.synapses;
        cons.reserve(ncons);

        const auto s = params_.ring_size;
        const auto group = gid/s;
        const auto group_start = s*group;
        const auto group_end = std::min(group_start+s, num_cells_);
        cell_gid_type src = gid==group_start? group_end-1: gid-1;
        cons.push_back(arb::cell_connection({src, "d"}, {"p"}, event_weight_, min_delay_*U::ms));

        // Used to pick source cell for a connection.
        std::uniform_int_distribution<cell_gid_type> dist(0, num_cells_-2);
        // Used to pick delay for a connection.
        std::uniform_real_distribution<float> delay_dist(0, 2*min_delay_);
        auto src_gen = std::mt19937(gid);
        for (unsigned i=1; i<ncons; ++i) {
            // Make a connection with weight 0.
            // The source is randomly picked, with no self connections.
            src = dist(src_gen);
            if (src==gid) ++src;
            const float delay = min_delay_+delay_dist(src_gen);
            cons.push_back(
                arb::cell_connection({src, "d"}, {"p"}, 0.f, delay*U::ms));
        }
        return cons;
    }

    // Return one event generator on the first cell of each ring.
    // This generates a single event that will kick start the spiking on the sub-ring.
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        if (gid%params_.ring_size == 0) {
            return {arb::explicit_generator_from_milliseconds({"p"}, event_weight_, std::vector{1.0})};
        } else {
            return {};
        }
    }

    std::vector<arb::probe_info> get_probes(cell_gid_type gid) const override {
        // Measure at the soma.
        arb::mlocation loc{0, 0.0};
        return {{cable_probe_membrane_voltage{loc}, "Um"}};
    }

    cell_size_type num_cells_;
    double min_delay_;
    float event_weight_;
    ring_params params_;

    arb::cable_cell_global_properties gprop;
};

struct cell_stats {
    using size_type = unsigned;
    size_type ncells = 0;
    size_type nbranch = 0;
    size_type ncomp = 0;

    cell_stats(arb::recipe& r) {
        int nranks, rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        ncells = r.num_cells();
        size_type cells_per_rank = ncells/nranks;
        size_type b = rank*cells_per_rank;
        size_type e = (rank==nranks-1)? ncells: (rank+1)*cells_per_rank;
        size_type nbranch_tmp = 0;
        for (size_type i=b; i<e; ++i) {
            auto c = arb::util::any_cast<arb::cable_cell>(r.get_cell_description(i));
            nbranch_tmp += c.morphology().num_branches();
            for (unsigned i = 0; i < c.morphology().num_branches(); ++i) {
                ncomp += c.morphology().branch_segments(i).size();
            }
        }
        MPI_Allreduce(&nbranch_tmp, &nbranch, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    }

    friend std::ostream& operator<<(std::ostream& o, const cell_stats& s) {
        return o << "cell stats: "
                 << s.ncells << " cells; "
                 << s.nbranch << " branches; "
                 << s.ncomp << " compartments; ";
    }
};

int main(int argc, char** argv) {
    try {
        bool root = true;

        auto params = read_options(argc, argv);

        arb::proc_allocation resources;
        resources.num_threads = arbenv::default_concurrency();
        resources.bind_threads = true;
        resources.bind_procs = true;

        arbenv::with_mpi guard(argc, argv, false);
        resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);
        auto context = arb::make_context(resources, MPI_COMM_WORLD);
        root = arb::rank(context) == 0;

        if (root) std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n"
                            << "threads:  " << num_threads(context) << "\n"
                            << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n"
                            << "ranks:    " << num_ranks(context) << "\n" << std::endl;

        arb::profile::meter_manager meters;
        meters.start(context);

        // Create an instance of our recipe.
        ring_recipe recipe(params);
        cell_stats stats(recipe);
        if (root) std::cout << stats << "\n";
        // Make decomposition
        auto decomp = arb::partition_load_balance(recipe, context);
        // Construct the model.
        arb::simulation sim(recipe, context, decomp);

        meters.checkpoint("model-init", context);

        // Run the simulation.
        if (root) sim.set_epoch_callback(arb::epoch_progress_bar());
        if (root) std::cout << "running simulation\n" << std::endl;
        sim.run(params.duration*U::ms, params.dt*U::ms);

        meters.checkpoint("model-run", context);

        auto ns = sim.num_spikes();

        // Write spikes to file
        if (root) std::cout << "\n" << ns << " spikes generated at rate of " << params.duration/ns << " ms between spikes\n";
        if (root) std::cout << arb::profile::make_meter_report(meters, context) << '\n';
    }
    catch (std::exception& e) {
        std::cerr << "exception caught in ring miniapp: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Helper used to interpolate in branch_cell.
template <typename T>
double interp(const std::array<T,2>& r, unsigned i, unsigned n) {
    double p = i * 1./(n-1);
    double r0 = r[0];
    double r1 = r[1];
    return r[0] + p*(r1-r0);
}

arb::segment_tree generate_morphology(arb::cell_gid_type gid, const cell_parameters& params) {
    arb::segment_tree tree;

    double soma_radius = 12.6157/2.0;
    int soma_tag = 1;
    tree.append(arb::mnpos, {0, 0,-soma_radius, soma_radius}, {0, 0, soma_radius, soma_radius}, soma_tag); // For area of 500 μm².

    std::vector<std::vector<unsigned>> levels;
    levels.push_back({0});

    // Standard mersenne_twister_engine seeded with gid.
    std::mt19937 gen(gid);
    std::uniform_real_distribution<double> dis(0, 1);

    double dend_radius = 0.5; // Diameter of 1 μm for each cable.
    int dend_tag = 3;

    double dist_from_soma = soma_radius;
    for (unsigned i=0; i<params.max_depth; ++i) {
        // Branch prob at this level.
        double bp = interp(params.branch_probs, i, params.max_depth);
        // Length at this level.
        double l = interp(params.lengths, i, params.max_depth);
        // Number of compartments at this level.
        unsigned nc = std::round(interp(params.compartments, i, params.max_depth));

        std::vector<unsigned> sec_ids;
        for (unsigned sec: levels[i]) {
            for (unsigned j=0; j<2; ++j) {
                if (dis(gen)<bp) {
                    auto z = dist_from_soma;
                    auto dz = l/nc;
                    auto p = sec;
                    for (unsigned k=1; k<nc; ++k) {
                        p = tree.append(p, {0,0,z+(k+1)*dz, dend_radius}, dend_tag);
                    }
                    sec_ids.push_back(p);
                }
            }
        }
        if (sec_ids.empty()) {
            break;
        }
        levels.push_back(sec_ids);

        dist_from_soma += l;
    }

    return tree;
}

arb::cable_cell complex_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    using arb::reg::tagged;
    using arb::reg::all;
    using arb::ls::location;
    using arb::ls::uniform;

    arb::segment_tree tree = generate_morphology(gid, params);

    auto rall  = arb::reg::all();
    auto soma = tagged(1);
    auto axon = tagged(2);
    auto dend = tagged(3);
    auto apic = tagged(4);
    auto cntr = location(0, 0.5);
    auto syns = arb::ls::uniform(rall, 0, params.synapses-2, gid);

    arb::decor decor;

    decor.paint(rall, arb::init_reversal_potential{"k",  -107.0*U::mV});
    decor.paint(rall, arb::init_reversal_potential{"na",   53.0*U::mV});

    decor.paint(soma, arb::axial_resistivity{133.577*U::Ohm*U::cm});
    decor.paint(soma, arb::membrane_capacitance{4.21567e-2*U::F/U::m2});

    decor.paint(dend, arb::axial_resistivity{68.355*U::Ohm*U::cm});
    decor.paint(dend, arb::membrane_capacitance{2.11248e-2*U::F/U::m2});

    decor.paint(soma, arb::density("pas/e=-76.4024", {{"g", 0.000119174}}));
    decor.paint(soma, arb::density("NaV",            {{"gbar", 0.0499779}}));
    decor.paint(soma, arb::density("SK",             {{"gbar", 0.000733676}}));
    decor.paint(soma, arb::density("Kv3_1",          {{"gbar", 0.186718}}));
    decor.paint(soma, arb::density("Ca_HVA",         {{"gbar", 9.96973e-05}}));
    decor.paint(soma, arb::density("Ca_LVA",         {{"gbar", 0.00344495}}));
    decor.paint(soma, arb::density("CaDynamics",     {{"gamma", 0.0177038}, {"decay", 42.2507}}));
    decor.paint(soma, arb::density("Ih",             {{"gbar", 1.07608e-07}}));

    decor.paint(dend, arb::density("pas/e=-88.2554", {{"g", 9.57001e-05}}));
    decor.paint(dend, arb::density("NaV",            {{"gbar", 0.0472215}}));
    decor.paint(dend, arb::density("Kv3_1",          {{"gbar", 0.186859}}));
    decor.paint(dend, arb::density("Im_v2",          {{"gbar", 0.00132163}}));
    decor.paint(dend, arb::density("Ih",             {{"gbar", 9.18815e-06}}));

    decor.place(cntr, arb::synapse("expsyn"), "p");
    if (params.synapses>1) {
        decor.place(syns, arb::synapse("expsyn"), "s");
    }

    decor.place(cntr, arb::threshold_detector{-20.0*U::mV}, "d");

    decor.set_default(arb::cv_policy_every_segment());

    return {arb::morphology(tree), decor};
}

arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    using arb::reg::tagged;

    arb::segment_tree tree = generate_morphology(gid, params);

    auto soma = tagged(1);
    auto dnds = join(tagged(3), tagged(4));
    auto syns = arb::ls::uniform(arb::reg::all(), 0, params.synapses-2, gid);

    arb::decor decor;

    decor.paint(soma, arb::density{"hh"});
    decor.paint(dnds, arb::density{"pas"});
    decor.set_default(arb::axial_resistivity{100*U::Ohm*U::cm}); // [Ω·cm]

    // Add spike threshold detector at the soma.
    decor.place(arb::mlocation{0,0}, arb::threshold_detector{10*U::mV}, "d");

    // Add a synapse to proximal end of first dendrite.
    decor.place(arb::mlocation{1, 0}, arb::synapse{"expsyn"}, "p");

    // Add additional synapses that will not be connected to anything.
    if (params.synapses>1) {
        decor.place(syns, arb::synapse{"expsyn"}, "s");
    }

    // Make a CV between every sample in the sample tree.
    decor.set_default(arb::cv_policy_every_segment());

    return {arb::morphology(tree), decor};
}

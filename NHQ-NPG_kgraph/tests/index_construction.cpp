#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <string>
#include <omp.h>
#include <chrono>

#include <thread>

#include <atomic>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

using namespace std;

int main(int argc, char **argv)
{
    // Get number of WH threads and use that number of threads for the index construction
    unsigned int nthreads = std::thread::hardware_concurrency();
    omp_set_num_threads(nthreads);

    // Prepare thread monitoring
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Parameters
    std::string path_database_vectors;
    std::string path_database_attributes;
    std::string path_index;
    int K;
	int L;
	int iter;
	int S;
	int R;
	int Range;
	int PL;
	float B;
	float M;

    // Parse arguments
    if (argc != 13) {
        fprintf(stderr, "Usage: %s <path_database_vectors> <path_database_attributes> <path_index> <K> <L> <iter> <S> <R> <Range> <PL> <B> <M>\n", argv[0]);
        exit(1);
    }

    // Store parameters
    path_database_vectors = argv[1];
    path_database_attributes = argv[2];
    path_index = argv[3];
	K = atoi(argv[4]);
	L = atoi(argv[5]);
	iter = atoi(argv[6]);
	S = atoi(argv[7]);
	R = atoi(argv[8]);
	Range = atoi(argv[9]);
	PL = atoi(argv[10]);
	B = atof(argv[11]);
	M = atof(argv[12]);
	
	// Use as many threads as available
	omp_set_num_threads(nthreads);

	// Load database vectors
	float *database_vectors = NULL;
 	unsigned n_items, d;
	efanna2e::load_data(const_cast<char*>(path_database_vectors.c_str()), database_vectors, n_items, d);
 	database_vectors = efanna2e::data_align(database_vectors, n_items, d); //one must align the data before build

	// Load database attributes
	std::vector<int> database_attributes = read_one_int_per_line(path_database_attributes);
	assert(database_attributes.size() == n_items);

	// Transform database attributes into format required by NHQ
    std::vector<std::vector<std::string>> database_attributes_str;
    for (std::size_t i = 0; i < database_attributes.size(); ++i) {
        database_attributes_str.push_back({std::to_string(database_attributes[i])});
    }

	// Initialize and configure the NHQ-kgraph index
	efanna2e::IndexRandom init_index(d, n_items);
	efanna2e::IndexGraph nhq_index(d, n_items, efanna2e::L2, (efanna2e::Index *)(&init_index));
	efanna2e::Parameters paras;
	paras.Set<unsigned>("K", K);
	paras.Set<unsigned>("L", L);
	paras.Set<unsigned>("iter", iter);
	paras.Set<unsigned>("S", S);
	paras.Set<unsigned>("R", R);
	paras.Set<unsigned>("RANGE", Range);
	paras.Set<unsigned>("PL", PL);
	paras.Set<float>("B", B);
	paras.Set<float>("M", M);

	// Build the index (this part is timed)
	auto start_time = std::chrono::high_resolution_clock::now();	
	for (int i = 0; i < n_items; i++){
		nhq_index.AddAllNodeAttributes(database_attributes_str[i]);
	}
	nhq_index.Build(n_items, database_vectors, paras);
	auto end_time = std::chrono::high_resolution_clock::now();

    // Stop thread monitoring
    done = true;
    monitor.join();

	// Print statistics
	std::chrono::duration<double> diff = end_time - start_time;
	double duration = diff.count();
	printf("Maximum number of threads: %d\n", peak_threads.load()-1);   // Subtract 1 because of the monitoring thread
	printf("Index construction time: %.3f s\n", duration);
	peak_memory_footprint();

	// Save the index
	std::string index_path_model  = path_index + "_model";
	std::string index_path_attribute_table = path_index + "_attribute_table";
	nhq_index.Save(index_path_model.c_str());
	nhq_index.SaveAttributeTable(index_path_attribute_table.c_str());

	return 0;
}

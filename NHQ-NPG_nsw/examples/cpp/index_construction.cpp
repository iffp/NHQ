#include "n2/hnsw.h"

#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>

#include <thread>

#include <atomic>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

using namespace std;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);


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
    int M;
    int MaxM0;
	int NumThread = nthreads; // TODO: Use as many threads as available, maybe change later
    int efConstruction;

	// Parse arguments
	if (argc != 7) {
		fprintf(stderr, "Usage: %s <path_database_vectors> <path_database_attributes> <path_index> <M> <MaxM0> <efConstruction>\n", argv[0]);
		exit(1);
	}

	// Store parameters
	path_database_vectors = argv[1];
	path_database_attributes = argv[2];
	path_index = argv[3];
	M = atoi(argv[4]);
	MaxM0 = atoi(argv[5]);
	efConstruction = atoi(argv[6]);

	// Load database vectors
	vector<vector<float>> database_vectors = read_fvecs(path_database_vectors);
	int n_items = database_vectors.size();
	int d = database_vectors[0].size();

	// Load database attributes
	vector<int> database_attributes = read_one_int_per_line(path_database_attributes);
	assert(database_attributes.size() == n_items);

	// Transform database attributes into format required by NHQ
	std::vector<std::vector<std::string>> database_attributes_str;
	for (std::size_t i = 0; i < database_attributes.size(); ++i) {
		database_attributes_str.push_back({std::to_string(database_attributes[i])});
	}
	
	// Initialize and configure the NHQ index
    n2::Hnsw nhq_index(d, "L2");
	vector<pair<string, string>> configs = {{"M", to_string(M)}, {"MaxM0", to_string(MaxM0)}, {"NumThread", to_string(NumThread)}, {"efConstruction", to_string(efConstruction)}};
	nhq_index.SetConfigs(configs);

	// Construct index (timed)
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n_items; i++) {
		nhq_index.AddData(database_vectors[i]);	
	}
	for (int i = 0; i < n_items; i++) {
		nhq_index.AddAllNodeAttributes(database_attributes_str[i]);
	}
    nhq_index.Fit();
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

	// Save the index to file
	std::string index_path_model = path_index + "_model";
	std::string index_path_attribute_table = path_index + "_attribute_table";
    nhq_index.SaveModel(index_path_model);
    nhq_index.SaveAttributeTable(index_path_attribute_table);

	return 0;
}

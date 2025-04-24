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

#include "fanns_survey_helpers.cpp"

using namespace std;

int main(int argc, char **argv)
{
    // Get number of threads
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "Number of threads: " << nthreads << std::endl;

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

	// Print statistics
	std::chrono::duration<double> diff = end_time - start_time;
	double duration = diff.count();

	// Report statistics
	printf("Index construction time: %.3f s\n", duration);
    peak_memory_footprint();

	// Save the index to file
	std::string index_path_model = path_index + "_model";
	std::string index_path_attribute_table = path_index + "_attribute_table";
    nhq_index.SaveModel(index_path_model);
    nhq_index.SaveAttributeTable(index_path_attribute_table);

	return 0;
}

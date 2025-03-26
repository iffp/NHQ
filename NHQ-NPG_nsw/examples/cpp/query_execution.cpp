#include "n2/hnsw.h"

#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <sstream>
#include <fstream>

using namespace std;

int main(int argc, char **argv)
{
	// Get number of threads
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "Number of threads: " << nthreads << std::endl;

    // Parameters
    std::string path_query_vectors;
    std::string path_query_attributes;
    std::string path_groundtruth;
    std::string path_index;
    int k;
    int weight_search;
	int ef_search;

	// Check if the number of arguments is correct
    if (argc != 8)
    {
		fprintf(stderr, "Usage: %s <path_query_vectors> <path_query_attributes> <path_groundtruth> <path_index> <k> <weight_search> <ef_search>\n", argv[0]);
		exit(1);
    }

	// Read command line arguments
	path_query_vectors = argv[1];
	path_query_attributes = argv[2];
	path_groundtruth = argv[3];
	path_index = argv[4];
	k = atoi(argv[5]);
	weight_search = atoi(argv[6]);
	ef_search = atoi(argv[7]);

	// Read query vectors
	vector<vector<float>> query_vectors = read_fvecs(path_query_vectors);
	n_queries = query_vectors.size();
	d = query_vectors[0].size();

	// Read query attributes
	vector<int> query_attributes = read_one_int_per_line(path_query_attributes);
	assert(query_attributes.size() == n_queries);

	// Transform query attributes into format required by NHQ
	std::vector<std::vector<std::string>> query_attributes_str;
	for (std::size_t i = 0; i < input.size(); ++i) {
		query_attributes_str.push_back({std::to_string(query_attributes[i])});
	}

	// Read groundtruth
	vector<vector<int>> groundtruth = read_ivecs(path_groundtruth);
	assert(groundtruth.size() == n_queries);

    // Truncate ground-truth to at most k items
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

	// Load NHQ index
    n2::Hnsw index;
	std::string index_path_model = path_index + "_model";
	std::string index_path_attribute_table = path_index + "_attribute_table";
    index.LoadModel(index_path_model);
    index.LoadAttributeTable(index_path_attribute_table);

	// Configure search parameters and prepare data structures
    vector<pair<string, string>> configs = {{"weight_search", to_string(weight_search)}};
    index.SetConfigs(configs);
    vector<vector<pair<int, float>>> result(n_queries);

	// Perform search query by query (timed)
	auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < n_queries; i++)
	{
		index.SearchByVector_new(query_vectors[i], query_attributes_str[i], k, ef_search, result[i]);
	}
	auto end_time = chrono::high_resolution_clock::now();

	// Compute search time
	chrono::duration<double> time_diff = end_time - start_time;
	double query_execution_time = time_diff.count();

	// Compute recall
	size_t match_count = 0;
	size_t total_count = 0;
	for (int i = 0; i < n_queries; i++){
		int n_valid_neighbors = min(k, (int)groundtruth[i].size());
		vector<int> groundtruth_q = groundtruth[i];
		vector<int> result_q;
		for (int j = 0; j < k; j++){
			result_q.push_back(stoi(result[i][j].first));
		}
		sort(groundtruth_q.begin(), groundtruth_q.end());
		sort(result_q.begin(), result_q.end());
		vector<int> intersection;
		set_intersection(groundtruth_q.begin(), groundtruth_q.end(), result_q.begin(), result_q.end(), back_inserter(intersection));	
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}
	
	// Report results	
	double recall = (double)match_count / total_count;
	double qps = n_queries / query_execution_time;
	peak_memory_footprint();
	printf("Queries per second: %.3f\n", qps);
	printf("Recall: %.3f\n", recall);
}

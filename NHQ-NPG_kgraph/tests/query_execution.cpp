#include <chrono>
#include <thread>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

#include <atomic>
#include <omp.h>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

using namespace std;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv){
    // Restrict number of threads to 1 for query execution
    omp_set_num_threads(1);

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Parameters
    std::string path_database_vectors;
    std::string path_query_vectors;
    std::string path_query_attributes;
    std::string path_groundtruth;
    std::string path_index;
    int k;
    int weight_search;
    int L_search;

    // Check if the number of arguments is correct
    if (argc != 9)
    {
        fprintf(stderr, "Usage: %s <path_database_vectors> <path_query_vectors> <path_query_attributes> <path_groundtruth> <path_index> <k> <weight_search> <L_search>\n", argv[0]);
        exit(1);
    }

    // Read command line arguments
	path_database_vectors = argv[1];
    path_query_vectors = argv[2];
    path_query_attributes = argv[3];
    path_groundtruth = argv[4];
    path_index = argv[5];
    k = atoi(argv[6]);
    weight_search = atoi(argv[7]);
	L_search = atoi(argv[8]);

	// Setting seed
	unsigned seed = 161803398;
	srand(seed);

	// Read database vectors
	unsigned n_items, d;
	float *database_vectors = nullptr;
	efanna2e::load_data(const_cast<char*>(path_database_vectors.c_str()), database_vectors, n_items, d);
	database_vectors = efanna2e::data_align(database_vectors, n_items, d);

	// Read query vectors
	unsigned n_queries, d2;
	float *query_vectors = nullptr;
	efanna2e::load_data(const_cast<char*>(path_query_vectors.c_str()), query_vectors, n_queries, d2);
	query_vectors = efanna2e::data_align(query_vectors, n_queries, d2);
	assert(d == d2);

	// Read query attributes
	vector<int> query_attributes = read_one_int_per_line(path_query_attributes);
	assert(query_attributes.size() == n_queries);

    // Transform query attributes into format required by NHQ
    std::vector<std::vector<std::string>> query_attributes_str;
    for (std::size_t i = 0; i < query_attributes.size(); ++i) {
        query_attributes_str.push_back({std::to_string(query_attributes[i])});
    }

	// Read ground-truth
	vector<vector<int>> groundtruth = read_ivecs(path_groundtruth);
	assert(groundtruth.size() == n_queries);

	// Truncate ground-truth to at most k items
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

	// Load NHQ index
	efanna2e::IndexRandom init_index(d, n_items);
	efanna2e::IndexGraph nhq_index(d, n_items, efanna2e::FAST_L2, (efanna2e::Index *)(&init_index));
    std::string index_path_model = path_index + "_model";
    std::string index_path_attribute_table = path_index + "_attribute_table";
	nhq_index.Load(index_path_model.c_str());
	nhq_index.LoadAttributeTable(index_path_attribute_table.c_str());

	// Perform the Optimizations	
	// TODO: Should this be timed as well?
	// NOTE: Doesn't work if we add this in the index construction
	nhq_index.OptimizeGraph(database_vectors);

	// Prepare search parameters
	efanna2e::Parameters paras;
	paras.Set<unsigned>("L_search", L_search);
	paras.Set<float>("weight_search", weight_search);

	// Prepare results
	std::vector<std::vector<unsigned>> result(n_queries);
	for (unsigned i = 0; i < n_queries; i++){
		result[i].resize(k);
	}

	// Perform the search (this is timed)
	auto start_time = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < n_queries; i++)
	{
		nhq_index.SearchWithOptGraph(query_attributes_str[i], query_vectors + i * d, k, paras, result[i].data());
	}
	auto end_time = std::chrono::high_resolution_clock::now();

    // Stop thread count monitoring
    done = true;
    monitor.join();

	// Compute search time
	std::chrono::duration<double> diff = end_time - start_time;
	double query_execution_time = diff.count();

	// Compute recall
	size_t match_count = 0;
	size_t total_count = 0;
	for (int i = 0; i < n_queries; i++){
		int n_valid_neighbors = min(k, (int)groundtruth[i].size());
		vector<int> groundtruth_q = groundtruth[i];
		vector<int> result_q;
		for (int j = 0; j < k; j++){
			result_q.push_back(result[i][j]);
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
	printf("Maximum number of threads: %d\n", peak_threads.load()-1);   // Subtract 1 because of the monitoring thread
	peak_memory_footprint();
	printf("Queries per second: %.3f\n", qps);
	printf("Recall: %.3f\n", recall);
	
	return 0;
}

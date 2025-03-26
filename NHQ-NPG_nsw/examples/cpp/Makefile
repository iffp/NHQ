CXX ?= g++
CXXFLAGS += -O3 -march=native -std=c++11 -pthread -fPIC -fopenmp
CXXFLAGS += -I../../include/ -I../../third_party/spdlog/include/ -I../../third_party/googletest/googletest/ -I../../third_party/googletest/googletest/include/
LDFLAGS += -lpthread -L../../build/lib/static -ln2 -fopenmp

all: index search index_construction query_execution

index: index.o
	$(CXX) -o $@  $? $(LDFLAGS)

search: search.o
	$(CXX) -o $@  $? $(LDFLAGS)

index_construction: index_construction.o
	$(CXX) -o $@  $? $(LDFLAGS)

query_execution: query_execution.o
	$(CXX) -o $@  $? $(LDFLAGS)

index.o: index.cpp
	$(CXX) $(CXXFLAGS) -c $?

search.o: search.cpp
	$(CXX) $(CXXFLAGS) -c $?

index_construction.o: index_construction.cpp
	$(CXX) $(CXXFLAGS) -c $?

query_execution.o: query_execution.cpp
	$(CXX) $(CXXFLAGS) -c $?

clean:
	rm -f *.o index search index_construction query_execution

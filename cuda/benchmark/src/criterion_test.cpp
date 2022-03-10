#include <criterion/criterion.hpp>

BENCHMARK(Simple_test)
{
  SETUP_BENCHMARK(
    const auto size = 100;
    std::vector<int> vec(size, 0); // vector of size 100
  )
 
  // Code to be benchmarked
  // merge_sort(vec.begin(), vec.end(), std::less<int>(), size);
  
  TEARDOWN_BENCHMARK(
    vec.clear();
  )
}

CRITERION_BENCHMARK_MAIN()
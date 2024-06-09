#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>

using namespace std::literals;

// BackTracking


int backtrack(STATE curr, OTHER_ARGUMENTS...) {
  if (BASE_CASE) {
    // modify the answer
    return 0;
  }
  int ans = 0;
  for (ITERATE_OVER_INPUT) {
    // modify the current state
    ans += backtrack(curr, OTHER_ARGUMENTS...)
    // undo the modification of the current state
  }

  return ans;
}


int main() {

  return 0;
}

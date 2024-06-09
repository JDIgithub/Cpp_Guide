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

class STATE{};
STATE STATE_FOR_WHOLE_INPUT;
bool BASE_CASE;


// Dynamic Programming Top-Down Memoization

std::unordered_map<STATE, int> memo;

int fn(std::vector<int>& arr) {
  return dp(STATE_FOR_WHOLE_INPUT, arr);
}

int dp(STATE, std::vector<int>& arr) {
  if (BASE_CASE) {
    return 0;
  }
  if (memo.find(STATE) != memo.end()) {
    return memo[STATE];
  }

  int ans = RECURRENCE_RELATION(STATE);
  memo[STATE] = ans;
  return ans;
}


int main() {

  return 0;
}

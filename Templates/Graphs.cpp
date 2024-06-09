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



int START_NODE {0};
// Graph DFS Recursive
// For the graph templates, assume the nodes are numbered from 0 to n - 1 and the graph is given as an adjacency list. 
// Depending on the problem, you may need to convert the input into an equivalent adjacency list before using the templates.

std::unordered_set<int> seen;

int fn(std::vector<std::vector<int>>& graph) {
  seen.insert(START_NODE);
  return dfs(START_NODE, graph);
}

int dfs(int node, std::vector<std::vector<int>>& graph) {
  int ans = 0;
  // do some logic
  for (int neighbor: graph[node]) {
    if (seen.find(neighbor) == seen.end()) {
      seen.insert(neighbor);
      ans += dfs(neighbor, graph);
    }
  }
  return ans;
}

// Graph: DFS Iterative
int fn(std::vector<std::vector<int>>& graph) {
  std::stack<int> stack;
  std::unordered_set<int> seen;
  stack.push(START_NODE);
  seen.insert(START_NODE);
  int ans = 0;

  while (!stack.empty()) {
    int node = stack.top();
    stack.pop();
    // do some logic
    for (int neighbor: graph[node]) {
      if (seen.find(neighbor) == seen.end()) {
        seen.insert(neighbor);
        stack.push(neighbor);
      }
    }
  }
}

// Graph BFS

int fn(std::vector<std::vector<int>>& graph) {
  std::queue<int> queue;
  std::unordered_set<int> seen;
  queue.push(START_NODE);
  seen.insert(START_NODE);
  int ans = 0;

  while (!queue.empty()) {
    int node = queue.front();
    queue.pop();
    // do some logic
    for (int neighbor: graph[node]) {
      if (seen.find(neighbor) == seen.end()) {
        seen.insert(neighbor);
        queue.push(neighbor);
      }
    }
  }
}


int main() {

  return 0;
}

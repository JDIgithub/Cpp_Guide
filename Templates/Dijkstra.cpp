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

int n;
int source{0};


std::vector<int> distances(n, INT_MAX);
distances[source] = 0;

std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> heap;
heap.push({0, source});

while (!heap.empty()) {
  int currDist = heap.top().first;
  int node = heap.top().second;
  heap.pop();
   
  if (currDist > distances[node]) {
    continue;
  }
    
  for (std::pair<int, int> edge: graph[node]) {
    int nei = edge.first;
    int weight = edge.second;
    int dist = currDist + weight;
        
    if (dist < distances[nei]) {
      distances[nei] = dist;
      heap.push({dist, nei});
    }
  }
}



int main() {

    
 
  return 0;
}

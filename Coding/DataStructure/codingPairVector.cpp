#include <iostream>
#include <vector>
#include <algorithm>

std::vector<std::pair<int, int>> mergeIntervals(const std::vector<std::pair<int, int>>& intervals) {
  
  if (intervals.empty()) return {};
  // Sort intervals based on the starting point
  std::sort(intervals.begin(), intervals.end());
  std::vector<std::pair<int, int>> merged;
  merged.push_back(intervals[0]);

  for (const auto& interval : intervals) {
    auto& last = merged.back(); // Reference so if we change last it will change merged

    if (last.second >= interval.first) { // Overlapping intervals, merge them
      last.second = std::max(last.second, interval.second); // Changing last element of merged vector
    } else {
      merged.push_back(interval); // Non-overlapping interval, add next element to the merged vector
    }
  }
  return merged;
}

int main() {
  
  std::vector<std::pair<int, int>> intervals = {{1,3}, {2,6}, {8,10}, {15,18}};
  auto mergedIntervals = mergeIntervals(intervals);

  for (const auto& interval : mergedIntervals) {
    std::cout << "{" << interval.first << "," << interval.second << "} ";
  }
  return 0;
}





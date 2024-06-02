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

/* 198. House Robber

You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected
and it will automatically contact the police if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.


Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 400

*/


class Semaphore {

std::mutex mtx;
std::condition_variable cv;
int counter{0};
int max_counter{1};

public:
// Increments the counter
void release(){
  // Use a mutex for thread safety
  std::lock_guard<std::mutex> lock(mtx);
  std::cout << "Adding one item " << std::endl; 
  if(counter < max_counter){
    // Increment
    ++counter;
    count();
  }
  // Use a condition variable to coordinate threads
  cv.notify_all(); 
}

// Decrements the counter
void acquire(){
  // Use a mutex for thread safety
  std::unique_lock<std::mutex> lock(mtx);
  std::cout << "Removing one item " << std::endl;
  // Make sure there is at least one in the counter
  while(counter == 0){ cv.wait(lock);}
  // Remove 
  --counter;
  count();
}

void count() const {
  std::cout << "Value of counter: ";
  std::cout << counter << std::endl;
}


};



int main(){

  Semaphore sem;

  auto insert = [&sem](){
    sem.release();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  };

  auto relinquish = [&sem](){
    sem.acquire();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  };

  std::vector<std::thread> tasks;

  for(int i = 0; i < 2; i++){ 
    tasks.push_back(std::thread(insert));
  }

  for(int i = 0; i < 4; i++){ 
    tasks.push_back(std::thread(relinquish));
  }

  for(int i = 0; i < 3; i++){ 
    tasks.push_back(std::thread(insert));
  }

  for(auto& task: tasks) task.join();

  return 0;
}
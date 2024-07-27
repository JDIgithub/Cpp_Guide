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
#include <csignal>
#include <optional>
#include <fstream>

using namespace std;

int main() {
  // Seed with a real random value, if available
  std::random_device rd;

  // Choose a random number between 1 and 50
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(1, 50);
  std::uniform_int_distribution<> dis2(1, 12);
  // Generate and print a random number
  std::vector<int> winningNumbers;
  std::vector<int> winningNumbers2;

  for(int i = 0; i < 5;){
    int freshNumber = dis(gen);
    if(std::find(winningNumbers.begin(),winningNumbers.end(),freshNumber) == winningNumbers.end()){
      winningNumbers.push_back(freshNumber);
      i++;
    }
  }
  for(int i = 0; i < 2;){
    int freshNumber = dis2(gen);
    if(std::find(winningNumbers2.begin(),winningNumbers2.end(),freshNumber) == winningNumbers2.end()){
      winningNumbers2.push_back(freshNumber);
      i++;
    }
  }

  std::cout << "Good Luck: ";
  for(auto num:winningNumbers){
    std::cout << num << " ";
  }
  std::cout << ": ";

  for(auto num:winningNumbers2){
    std::cout << num << " ";
  }



  return 0;
}





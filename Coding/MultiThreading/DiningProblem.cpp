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

using namespace std;
using namespace std::literals;

//  Dining Philosophers
/*

5 philosophers sit at a round table which has 5 forks on it
A philosophers have a fork at each side of them
A philosopher can only eat i they can pick up both forks
If a philosopher picks up the fork on their right, that prevents the next philosopher from picking up their left fork

Deadlock scenario:
  All the philosophers pick up their left fork at the same time
  They wait to pick up their right fork which is taken -> All the philosophers are waiting for their neighbour -> Deadlock

*/

constexpr int nforks = 5;
constexpr int nphil = 5;
std::string names[nphil] = {"A", "B", "C", "D", "E"};
constexpr auto thinkTime = 2s;
constexpr auto eatTime = 1s;
// Keep track of how many times a philosopher is able to eat
int mouthfuls[nphil] = {0};

// A philosopher thread can only pick up a fork if it can lock the corresponding mutex
std::mutex fork_mutex[nforks];
// Mutex to protect output
std::mutex printMutex;

void print(int n, const std::string& str, int forkNO){
  std::lock_guard<std::mutex> printLock(printMutex);
  std::cout << "Philosopher " << names[n] << str << forkNO << '\n';
}

void print(int n, const std::string& str){
  std::lock_guard<std::mutex> printLock(printMutex);
  std::cout << "Philosopher " << names[n] << str << '\n';
}

// Thread which represents a dining philosopher
void dine(int nphilo){
  // Philosopher A has fork 0 on their left and fork 1 on their right
  // Each philosopher must pick up their left for first
  int lfork = nphilo;
  int rfork = (nphilo+1) % nforks;

  // To reach hierarchical mutex ordering
  if(lfork > rfork){
    std::swap(lfork,rfork);  
  }

  print(nphilo, "\'s left fork is number", lfork);
  print(nphilo, "\'s right fork is number", rfork);
  print(nphilo, " is thinking...");
  std::this_thread::sleep_for(thinkTime);

  // Make an attempt to eat
  print(nphilo, " reaches for for number ", lfork);

  // Try to pick up the left fork
  fork_mutex[lfork].lock();
  print(nphilo, " picks up fork ", lfork);
  print(nphilo, " is thinking...");
  std::this_thread::sleep_for(thinkTime);

  // Right fork
  print(nphilo, " reaches for for number ", rfork);
  // Try to pick up the right fork
  fork_mutex[rfork].lock();
  print(nphilo, " picks up fork ", rfork);
  print(nphilo, " is eating...");
  std::this_thread::sleep_for(eatTime);
  mouthfuls[nphilo]++;
  print(nphilo, " puts down fork ", lfork);
  print(nphilo, " puts down fork ", lfork);
  fork_mutex[lfork].unlock();
  fork_mutex[rfork].unlock();
  print(nphilo, " is thinking...");
  std::this_thread::sleep_for(thinkTime);

}


int main(){

  std::vector<std::thread> philosophers;

  for(int i = 0; i < nphil; i++){
    philosophers.push_back(std::move(std::thread(dine,i)));
  }
  for(auto& philosopher: philosophers) philosopher.join();
  
  for(int i = 0; i < nphil; i++){
    std::cout << "Philosopher " << names[i];
    std::cout << " had " << mouthfuls[i] << " mouthfuls\n";
  }
  
  return 0;
}
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


using namespace std;


// Atomic flag to indicate an interrupt has occurred
std::atomic<bool> interruptFlag(false);

// Interrupt handler function
void interruptHandler(int signal) {
    interruptFlag.store(true);
}

// Simulated interrupt generator
void generateInterrupt() {
    // Simulate an interrupt after 2 seconds
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::raise(SIGINT);
}


int main() {
  // Register signal handler
  std::signal(SIGINT, interruptHandler);

  // Start interrupt generator in a separate thread
  std::thread interruptThread(generateInterrupt);

  std::cout << "Main program running..." << std::endl;

  // Main program loop
  while (true) {
    if (interruptFlag.load()) {
      std::cout << "Interrupt occurred!" << std::endl;
      interruptFlag.store(false);
      // Handle interrupt (e.g., perform some task)
    }

    // Simulate main program doing some work
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << "Main program working..." << std::endl;
  }

  // Join the interrupt thread (this point is never reached in this example)
  interruptThread.join();

  return 0;
}






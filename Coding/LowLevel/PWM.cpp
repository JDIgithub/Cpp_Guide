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


// Simulate setting the PWM duty cycle (0-255)
void setPWMDutyCycle(int dutyCycle) {
  if (dutyCycle < 0) dutyCycle = 0;
  if (dutyCycle > 255) dutyCycle = 255;

  // Convert duty cycle to a percentage (0-100%)
  float dutyPercent = (dutyCycle / 255.0) * 100.0;
  
  // Simulate PWM signal by toggling LED state
  for (int i = 0; i < 100; ++i) {
    if (i < dutyPercent) {
      std::cout << "LED ON" << endl;
    } else {
      std::cout << "LED OFF" << endl;
    }
    std::this_thread::sleep_for(chrono::milliseconds(10)); // Simulate the PWM frequency
  }
}


int main() {
    int dutyCycle;

    std::cout << "Enter PWM duty cycle (0-255): ";
    std::cin >> dutyCycle;

    std::cout << "Simulating PWM signal with duty cycle: " << dutyCycle << "%" << endl;
    setPWMDutyCycle(dutyCycle);

    return 0;
}




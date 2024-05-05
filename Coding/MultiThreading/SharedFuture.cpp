#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
using namespace std;

// 13. Roman to Integer

/*

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000


Roman numerals are usually written largest to smallest from left to right. 
However, the numeral for four is not IIII. Instead, the number four is written as IV. 
Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX.
There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.


Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].

*/
using namespace std::literals;
using namespace std::chrono;

std::mutex mut;

void produce(std::promise<int>& prom){
 
  // Produce result
  int x = 42;
  std::this_thread::sleep_for(2s);
  
  // Store the result in the shared state
  std::cout << "Promise sets shared state to " << x << '\n';
  prom.set_value(x);
}

void consume(std::shared_future<int>& fut){

  // Get the result from the shared state
  // Waits till the producer produces the value with std::promise
  std::cout << "Thread " << std::this_thread::get_id() << " calling get()\n";
  int x = fut.get();
  std::lock_guard<std::mutex> lck_guard(mut);
  std::cout << "Thread " << std::this_thread::get_id() << " has answer " << x << "\n";
}

int main(){

  // Create an std::promise object
  std::promise<int> prom;
  // Get the associated future
  std::shared_future<int> shared_fut1 = prom.get_future();
  std::shared_future<int> shared_fut2 = shared_fut1;
  std::thread producer(produce,std::ref(prom));
  std::thread consumer(consume,std::ref(shared_fut1));
  std::thread consumer2(consume,std::ref(shared_fut2));

  producer.join();
  consumer.join();
  consumer2.join();

  return 0;
}



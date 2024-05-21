#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
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
#include <stack>
#include <list>
#include <atomic>

using namespace std;

// 242. Valid Anagram
/*

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
 

Constraints:

1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
 

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

*/

bool isAnagram(string s, string t) {
   return 0;     
}


using namespace std::literals;


int task(){

  std::cout << "Executing task() in thread with ID: ";
  std::cout << std::this_thread::get_id() << "\n";
  std::this_thread::sleep_for(5s);
  std::cout << "Returning from task()\n";

  return 42; 
}

void func(const std::string& option = "default"){

  std::future<int> result;
  if(option == "async"s){
    result = std::async(std::launch::async, task);
  } else if (option == "deferred"s){
    result = std::async(std::launch::deferred, task);
  } else { 
    result = std::async(task);
  }

  std::cout << "Calling Async with option \"" << option << "\"\n";
  std::this_thread::sleep_for(2s);
  std::cout << "Calling get() \n";
  std::cout << "Task result: " << result.get() << "\n";

}

int main(){

  std::cout << "In main thread with ID: " << std::this_thread::get_id() << "\n";
  
  func("async");
  func("deferred");   // Deferred runs in the same thread as the main thread here but 
                      // but task is not running before get() is called.
  func("default");

  std::cout << "Exiting main()\n";

  return 0;
}







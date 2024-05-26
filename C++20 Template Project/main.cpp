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
#include <random>
#include <numeric>
#include <execution>


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


std::timed_mutex t_mutex;

void task1(){
  std::cout << "Task1 trying to lock the mutex \n";
  std::lock_guard<std::timed_mutex> lck_guard(t_mutex);
  std::cout << "Task1 locks the mutex \n";
  std::this_thread::sleep_for(5s);
  std::cout << "Task1 unlocking the mutex \n";
}

void task2(){
  std::this_thread::sleep_for(500ms);
  std::cout << "Task2 trying to lock the mutex \n";
  std::unique_lock<std::timed_mutex> unique_lck(t_mutex, std::defer_lock);  // Defer lock so we could lock it ourselves

  // Try for 1 second to lock the mutex
  while(!unique_lck.try_lock_for(1s)){
    // Returned false
    std::cout << "Task2 could not lock the mutex \n"; 
  }
  // Returned true -> mutex is now locked
  std::cout << "Task2 locked the mutex \n";
}



int main(){

  std::thread thr1(task1);
  std::thread thr2(task2);
  
  thr1.join();
  thr2.join();

  return 0;
}







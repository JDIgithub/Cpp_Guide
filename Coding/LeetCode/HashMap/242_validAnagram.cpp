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

bool isAnagram(std::string s, std::string t) {

  if(s.size() != t.size()) return false;

  std::unordered_map<char,int> wordDictionary;

  for(int i = 0; i < s.size(); i++){
    wordDictionary[s[i]]++;
  }

  for(int i = 0; i < t.size(); i++){

    if(wordDictionary[t[i]] == 0){
      return false;
    } else {
      wordDictionary[t[i]]--;
    }
  }
  return true;     
}


int main(){

  std::cout << isAnagram("ab", "ba");
  
  return 0;
}







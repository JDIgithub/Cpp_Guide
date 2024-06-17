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

using namespace std;


/* 49 Group Anagrams

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Example 2:

Input: strs = [""]
Output: [[""]]
Example 3:

Input: strs = ["a"]
Output: [["a"]]
 

Constraints:

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.


*/

std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {

  std::vector<std::vector<std::string>> result;
  std::unordered_map<std::string,int> anagramGroups;

  std::vector<std::string> anagramGroup;
  int groupsCount = 0;

  for(auto& str : strs){

    std::string temp = str; 
    std::sort(temp.begin(),temp.end());
    
    if(anagramGroups.find(temp) == anagramGroups.end()){
      anagramGroups[temp] = groupsCount++; 
      result.push_back({str});
    } else {
      result[anagramGroups[temp]].push_back(str);
    }
  }
  return result;
}


// Sligthly different solution using vector of string in the hashmap
// Do not need to manage indices and group count but storing vectors of string...
vector<vector<string>> groupAnagrams2(vector<string>& strs) {
  
  unordered_map<string, vector<string>> anagramMap;

  for (string s : strs) {
    string key = s;
    sort(key.begin(), key.end());
    anagramMap[key].push_back(s);
  }

  vector<vector<string>> result;
  for (auto& pair : anagramMap) {
    result.push_back(pair.second);
  }

    return result;
}



int main() {

  std::vector<std::string> strs = { "eat","tea","tan","ate","nat","bat" };



  auto xx = groupAnagrams(strs);

  //std::cout << xx;



  return 0;


}





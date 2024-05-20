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

// 14. Longest Common Prefix
/*

Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
 

Constraints:

1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.
  
*/


std::string longestCommonPrefix(std::vector<std::string>& strings) {

  if(strings.size() == 0) return {};
  if(strings.size() == 1) return strings[0];  
  std::string commonPrefix;
  
  for(int i = 0; i < strings[0].size(); i++) {
    for(int j = 1; j < strings.size();j++){
      if(strings[0][i] != strings[j][i]){
        return commonPrefix;
      }
    }
    commonPrefix.push_back(strings[0][i]);
  }
  return commonPrefix;
}


int main(){

  std::vector<std::string> strings = {{"flower"},{"flow"},{"flight"}};
  
  std::cout << longestCommonPrefix(strings);


  return 0;
}







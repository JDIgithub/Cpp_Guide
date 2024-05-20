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

// 28. Index of the first occurence
/*

Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

 

Example 1:

Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
Example 2:

Input: haystack = "leetcode", needle = "leeto"
Output: -1
Explanation: "leeto" did not occur in "leetcode", so we return -1.
 

Constraints:

1 <= haystack.length, needle.length <= 104
haystack and needle consist of only lowercase English characters.
  
*/

int strStr(string haystack, string needle) {
        
  if(haystack.empty() || needle.empty()) return -1;
  int nIndex = 0;
  int hIndex = 0;
  int needleFound = -1;
  
  while(hIndex < haystack.size()){
    
    if(nIndex == needle.size()) return needleFound;

    if(haystack[hIndex] == needle[nIndex]){
      if(needleFound == -1){
        needleFound = hIndex;
      }
      nIndex++;
    } else {
      hIndex = hIndex - nIndex;
      nIndex = 0;
      needleFound = -1;
    }

    hIndex++;
  }
  if(nIndex != needle.size()) {
    return -1;
  } else {
    return needleFound;
  }
}

int strStrWithSubstrFunction(std::string haystack, std::string needle) {
  for (int i = 0; i <= haystack.length() - needle.length(); ++i) {
    if (haystack.substr(i, needle.length()) == needle) {
      return i;
    }
  }
  return -1;
}

int main(){

  std::string haystack = {"mississippi"};
  std::string needle = {"issip"};

  std::cout << strStr(haystack,needle);


  return 0;
}







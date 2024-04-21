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
#include <condition_variable>
#include <math.h>
using namespace std;

// 392. Is Subsequence

/*

Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters 
without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

Example 1:

Input: s = "abc", t = "ahbgdc"
Output: true
Example 2:

Input: s = "axc", t = "ahbgdc"
Output: false

*/

bool isSubsequence(string s, string t) {

  if(s.empty()) return true;  
  if(t.empty()) return false;
  
  int pointerA = 0;
  int pointerB = 0;

  while(pointerB <= (t.size()-1)){
    if(s[pointerA] == t[pointerB]){
      if(pointerA == (s.size()-1)) {return true; }
      pointerA++;
      pointerB++;
      
    } else {
      pointerB++;
    }
  }
  return false;
}



int main(){

  std::vector<int> nums {1,1,1,1};

  std::cout << isSubsequence("abc","ahbgdc");



  return 0;
}



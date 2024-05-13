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
#include <stack>


using namespace std;

// 205. Isomorphic Strings

/*

Given two strings s and t, determine if they are isomorphic.
Two strings s and t are isomorphic if the characters in s can be replaced to get t.
All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

Example 1:

Input: s = "egg", t = "add"
Output: true
Example 2:

Input: s = "foo", t = "bar"
Output: false
Example 3:

Input: s = "paper", t = "title"
Output: true
 
Constraints:

1 <= s.length <= 5 * 104
t.length == s.length
s and t consist of any valid ascii character.

*/

bool isIsomorphic(std::string s, std::string t) {

  if(s.empty() && t.empty()) return true;
  if(s.length() != t.length()) return false;
  
  std::unordered_map<char,char> checkS;
  std::unordered_map<char,char> checkT;
  checkS[s[0]] = t[0];
  checkT[t[0]] = s[0];

  for(int i = 1; i < s.length(); i++){
    if(checkS.find(s[i]) == checkS.end()){
      checkS[s[i]] = t[i];
    } else {
      if(checkS[s[i]] != t[i]){
        return false;
      }
    }
    if(checkT.find(t[i]) == checkT.end()){
      checkT[t[i]] = s[i];
    } else {
      if(checkT[t[i]] != s[i]){
        return false;
      }
    } 
  } 
  return true;
}



int main(){

  std::string s {"badc"};
  std::string t {"baba"};

  std::cout << isIsomorphic(s,t);

  return 0;
}



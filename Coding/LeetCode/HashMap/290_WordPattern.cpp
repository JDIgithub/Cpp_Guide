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

// 290. Word Pattern
/*

Given a pattern and a string s, find if s follows the same pattern.
Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in s.


Example 1:

Input: pattern = "abba", s = "dog cat cat dog"
Output: true

Example 2:

Input: pattern = "abba", s = "dog cat cat fish"
Output: false

Example 3:

Input: pattern = "aaaa", s = "dog cat cat dog"
Output: false
  
Constraints:

  1 <= pattern.length <= 300
  pattern contains only lower-case English letters.
  1 <= s.length <= 3000
  s contains only lowercase English letters and spaces ' '.
  s does not contain any leading or trailing spaces.
  All the words in s are separated by a single space.

*/

bool wordPattern(std::string pattern, std::string str) {

  // To solve this problem of checking if a string follows a given pattern with a bijection between characters in the pattern and words in the string
  // We need two hash maps (dictionaries). One map will store the mapping from pattern characters to words, and the other will store the mapping from words to pattern characters. 
  // This ensures that the mapping is bijective.

  std::unordered_map<char,std::string> patternCheck;
  std::unordered_map<std::string,char> wordCheck;      
  
  // Creating Words ---------------------
  std::istringstream iss(str);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word) {
    words.push_back(word);
  }
  // ------------------------------------

  if(words.size() != pattern.size()) return false;

  for(int i = 0; i < pattern.size(); i++){

    if( (patternCheck.find(pattern[i]) != patternCheck.end()) && (patternCheck[pattern[i]] != words[i]) ){ return false; }
    if( (wordCheck.find(words[i]) != wordCheck.end()) && (wordCheck[words[i]] != pattern[i]) ){ return false; }  

    patternCheck[pattern[i]] = words[i];
    wordCheck[words[i]] = pattern[i];
  }
  return true;
}



int main(){

  std::string str = {"dog dog dog dog"};
  std::string pattern = {"abba"};

  std::cout << wordPattern(pattern,str);


  return 0;
}







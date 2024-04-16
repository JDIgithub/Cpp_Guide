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

// 383. Ransom Note
/*

Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.

 

Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:

Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:

Input: ransomNote = "aa", magazine = "aab"
Output: true


*/

// with HashMap
bool canConstruct(string ransomNote, string magazine) {
        
  std::unordered_map<char,int> map;
  bool can = true;
  for(char c: magazine){
      map[c]++;           // hash map will init integer to 0 for the first entry so this works fine
  }

  for(char c: ransomNote){
    
    if(map.find(c) != map.end()){
      map[c]--;
    } else {
      return false;
    }
    

  }


  return true;
}

// with array

bool canConstruct(string ransomNote, string magazine) {
        
 // std::vector<int> letters(256,0);
  int letters[256]{0};  // Less memory and faster than vector


  for (char c : magazine) {
    letters[c]++;
  }

  for (char c : ransomNote) {
    if(letters[c] > 0){
      letters[c]--;
    } else {
      return false;
    }
  }
  return true;
}

// The fastest way
bool canConstruct(string ransomNote, string magazine) {
  int letters[256] = {0};
  for(int i = 0; i < magazine.size(); i++) {
    letters[magazine[i]]++;
  }
  for(int i = 0; i < ransomNote.size(); i++) {
    letters[ransomNote[i]]--;
  }
  for(int i = 0; i < 256; i++) {
    if(letters[i] < 0) {
      return false; 
    }
  }
  return true;
}

int main(){


  std::cout << canConstruct("aa","aab");



  return 0;
}



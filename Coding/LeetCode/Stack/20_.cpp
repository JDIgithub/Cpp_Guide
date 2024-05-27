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

using namespace std;

// 20. Valid Parantheses
/*

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "()[]{}"
Output: true
Example 3:

Input: s = "(]"
Output: false
 

Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
*/

bool isValid(std::string s) {

  if(s.empty()) return false;

  std::stack<char> parStack;

  for(char c: s){

    if(c == '(' || c == '[' || c =='{'){
      parStack.push(c);
    }
    if(c == ')'){
      if(parStack.empty()) { return false;}
      if (parStack.top() == '(') { parStack.pop(); 
      } else { return false;}
    } else if(c == '}'){
      if(parStack.empty()) { return false;}
      if (parStack.top() == '{') { parStack.pop(); 
      } else { return false;}
    } else if(c == ']'){
      if(parStack.empty()) { return false;}
      if (parStack.top() == '[') { parStack.pop(); 
      } else { return false;}
    } 
  }

  if(parStack.empty()) return true;
  return false;
}

// hash map to reduce repetetive code
bool isValidHM(std::string s) {
  if (s.empty()) return false;
  std::stack<char> parStack;
  std::unordered_map<char, char> matchingPair = {
    {')', '('},
    {'}', '{'},
    {']', '['}
  };
  for (char c : s) {
    if (matchingPair.find(c) != matchingPair.end()) {
      if (parStack.empty() || parStack.top() != matchingPair[c]) { return false; }
      parStack.pop();
    } else {
      parStack.push(c);
    }
  }
  return parStack.empty();
}

int main(){

  std::vector<int> nums{1,2,3,1};
  auto result = isValid("()");
  int pause = 0;
  
  return 0;
}
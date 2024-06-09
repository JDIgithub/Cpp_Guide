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

using namespace std::literals;

/* 17. Letter Combination of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. 
Return the answer in any order.
A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example 1:

  Input: digits = "23"
  Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

Example 2:

  Input: digits = ""
  Output: []

Example 3:

  Input: digits = "2"
  Output: ["a","b","c"]
 
Constraints:

0 <= digits.length <= 4
digits[i] is a digit in the range ['2', '9'].
*/

std::unordered_map<char,std::string> keyboard = {
  {'2', "abc"},
  {'3', "def"},
  {'4', "ghi"},
  {'5', "jkl"},
  {'6', "mno"},
  {'7', "pqrs"},
  {'8', "tuv"},
  {'9', "wxyz"}
};


void findCombinations(std::vector<std::string> &lettersToUse,std::vector<std::string>&result,std::string combination,int row){
  
  if(row == lettersToUse.size()){
    result.push_back(combination);
    return;
  }

  for(int column = 0; column <lettersToUse[row].size();column++){
    combination+=lettersToUse[row][column];
    findCombinations(lettersToUse,result,combination,row+1);
    combination.pop_back();
  }
}


std::vector<std::string> letterCombinations(std::string digits) {

  std::vector<std::string> lettersToUse;
  std::vector<std::string> result;
  if(digits.size()==0) return lettersToUse;
  for(char digit: digits){
    lettersToUse.push_back(keyboard[digit]);
  }
  int row = 0;
  std::string combination="";
  findCombinations(lettersToUse,result,combination,row);
  return result;
}


int main() {

  auto result = letterCombinations("23");


  return 0;
}

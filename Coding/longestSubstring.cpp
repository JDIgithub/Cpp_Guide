#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
using namespace std;

/*
LeetCode 3. Longest Substring Without Repeating Characters

Given a string s, find the length of the longest substring without repeating characters.

Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Example 2:
Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

Constraints:
0 <= s.length <= 5 * 10^4
s consists of English letters, digits, symbols, and spaces.
Approach:
A common approach to solve this problem is using the Sliding Window technique combined with a Hash Map to track characters and their indices in the string.
This allows us to skip characters immediately when we encounter a repeated character.

*/


int MylongestSubstring(const std::string& str){

  if(str.empty()) return 0;

  std::unordered_map<char,int> charIndx;
  int maxLength = 0; // Store the maximum length found.
  int start = 0; // Sliding window start index.
  
  for(int end = 0; end< str.size(); end++){

    if(charIndx.find(str[end]) != charIndx.end() && charIndx[str[end]] >= start){ 
      start = charIndx[str[end]] + 1; // Move the start of the window past this character's last occurrence.
    }
    charIndx[str[end]] = end; // Saves characeters and their idex into hash map. If char is already there just update his last index
    maxLength = std::max(maxLength, end - start + 1); // If length of actual windows is higher than maxlength then update it
  }

  return maxLength;
}

int longestSubstring(string s) {
  // 3x faster
  // int value of char as vector index and string index of that char as vector value
  // This way we can check if some letter was already use and where with O(1)
  // This direct indexing is faster because it avoids the overhead associated with hash computation and collision handling in the unordered map.
  // Vectors in C++ are contiguous in memory, which makes them cache-friendly. 
  // Accessing elements in a vector sequentially or via direct indexing (as done in the second algorithm) is more likely to benefit from CPU cache, 
  // reducing the memory access time. On the other hand, the elements in an unordered_map are not stored contiguously; accessing them might result in more cache misses, 
  // leading to slower performance.

  std::vector<int> mpp(256, -1);
  int left = 0, right = 0;
  int n = s.size();
  int len = 0;
  while (right < n) {
    if (mpp[s[right]] != -1){ left = max(mpp[s[right]] + 1, left); }
    mpp[s[right]] = right;
    len = max(len, right - left + 1);
    right++;
  }
  return len;
}

int main(){

  std::string str {"pwwkew"};
  std::cout << longestSubstring(str);

  return 0;
}


